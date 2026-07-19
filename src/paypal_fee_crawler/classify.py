"""Derive product-specific transaction fee rules from normalized PayPal tables.

The classifier works at the row level: a single PayPal table may contain several
independent payment products, and each relevant fee row becomes a separate
``TransactionFeeRule``.  Fixed-fee and international-surcharge schedules are kept
separate per product or product family so that an HTTP fee calculator can select
the schedule that applies to a given rule.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .models import (
    AmbiguousFeeRow,
    CoverageSummary,
    CurrencyConversion,
    DerivedFeeResult,
    Diagnostic,
    FeeComponent,
    FixedFeeSchedule,
    InternationalSurchargeSchedule,
    InternationalSurchargeScheduleEntry,
    Provenance,
    RateReference,
    ResolvedRate,
    Row,
    Source,
    Table,
    TransactionFeeRule,
    UnclassifiedFeeRow,
)
from .normalize import clean_text, normalize_decimal_string
from .pricing_tokens import CURRENCY_CODES

logger = logging.getLogger(__name__)

CLASSIFIER_VERSION = "rules-v1"
_CLASSIFIER_VERSION = CLASSIFIER_VERSION

# ---------------------------------------------------------------------------
# Language-aware product aliases.  More specific / longer aliases are listed
# first so that substring scoring naturally prefers them.
# ---------------------------------------------------------------------------

_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "paypal_checkout": (
        "paypal checkout",
        "paypal-zahlung",
        "paypal bezahlen",
        "paypal checkout-transaktionen",
        "betal med paypal",
        "betaling via paypal",
        "paypal betaling",
        "paypal betal",
        "betal med venmo",
        "betaling med venmo",
        "venmo",
        "pay with paypal",
        "paga con paypal",
        "betalen met paypal",
    ),
    "goods_and_services": (
        "sending and receiving money for goods and services",
        "geld für waren und dienstleistungen senden/empfangen",
        "geld für waren und dienstleistungen",
        "waren und dienstleistungen",
        "goods and services",
        "goods & services",
        "goods or services",
        "varer og tjenesteydelser",
        "varer og tjenester",
        "bienes y servicios",
        "beni e servizi",
        "goederen en diensten",
        "produkter och tjänster",
        "produkter og tjenester",
    ),
    "advanced_card_payments": (
        "advanced credit and debit card payments",
        "erweiterte kredit- und debitkartenzahlungen",
        "zahlungen mit kredit- und debitkarten mit erweiterten funktionen",
        "kredit- und debitkarten mit erweiterten funktionen",
        "advanced card",
        "erweiterte kartenzahlung",
        "kredit- og betalingskort",
        "kredit- og debitkort",
        "kredit- och debitkort",
        "kreditkort",
        "betalingskort",
        "kortbetalinger",
        "avancerede kortbetalinger",
        "credit and debit card",
        "credit/debit card",
        "tarjetas de crédito y débito",
        "tarjeta de crédito y débito",
        "pagos avanzados con tarjeta",
        "pagos con tarjeta de crédito y débito",
        "cartão de crédito e débito",
        "cartão de crédito",
        "paiement par carte bancaire avancé",
        "paiements par carte bancaire avancés",
        "carte bancaire avancée",
        "cartes bancaires avancées",
        "pagamento con carta di credito e debito avanzata",
        "pagamento avanzato con carta",
        "płatność kartą kredytową i debetową",
        "płatność kartą kredytową",
        "płatności kartą kredytową i debetową",
        "avancerade kortbetalningar",
        "creditcard- en debetcardbetalingen",
        "geavanceerde creditcard",
        "online betalen met creditcard",
        "e-terminal",
        "eterminal",
        "solution hébergée",
        "hosted solution",
        "solución alojada",
        "soluzione ospitata",
        "payments advanced",
        "payments pro",
        "virtual terminal",
        "additional risk",
        "risk factors",
        "chargeback protection",
        "fraud protection",
        "failure to implement",
        "express checkout",
        "foreign exchange",
        "fx as a service",
    ),
    "other_commercial": (
        "all other commercial transactions",
        "alle anderen geschäftlichen transaktionen",
        "sonstige gewerbliche transaktionen",
        "other commercial",
        "sonstige geschäftliche",
        "other commercial transactions",
        "commercial transactions",
        "erhvervsbetalinger",
        "erhverv",
        "øvrige erhvervsbetalinger",
        "alle andre erhvervsbetalinger",
        "andre erhvervsbetalinger",
        "forretningsbetalinger",
        "business payments",
        "øvrige forretningsbetalinger",
        "alle anderen",
        "andere commercial",
        "altre transazioni commerciali",
        "otras transacciones comerciales",
        "autres transactions commerciales",
        "andre forretningstransaksjoner",
        "pozostałe transakcje handlowe",
        "campaign",
        "store cash",
        "pyusd",
        "pay by bank",
        "ach",
    ),
    "alternative_payment_methods": (
        "alle anderen alternativen zahlungsmethoden",
        "alternative zahlungsmethode",
        "alternative zahlungsmethoden",
        "alternative payment method",
        "alternative payment methods",
        "all other alternative payment methods",
        "apm-transaktionsgebühren",
        "apm",
        "abm",
        "alternativ betalingsmetode",
        "alternative betalingsmetode",
        "alternative betaalmethode",
        "métodos de pago alternativos",
        "metodi di pagamento alternativi",
        "metodo di pagamento alternativo",
        "online bank transfer",
        "online bankoverførsel",
        "bankoverførsel",
        "bank transfer",
        "cash a check",
        "cheque",
        "check",
    ),
    "guest_checkout": (
        "zahlung eines nutzers unserer bedingungen für zahlungen ohne paypal-konto",
        "zahlungen ohne paypal-konto",
        "payments without a paypal account",
        "zahlung ohne paypal-konto",
        "guest checkout",
        "betalinger uden en paypal-konto",
        "uden en paypal-konto",
        "uden paypal-konto",
        "betal uden paypal",
        "betalning utan paypal-konto",
        "betalning utan paypal",
        "betaling uten paypal-konto",
        "betalen zonder paypal-account",
        "payer sans compte paypal",
        "pagar sin cuenta de paypal",
        "pagamento senza conto paypal",
        "płatność bez konta paypal",
    ),
    "invoice_pay_later": (
        "rechnungskauf mit ratepay",
        "ratepay",
        "invoice payments",
        "pay later",
        "ratenzahlungsangebote",
        "rechnungskauf",
        "faktura",
        "fakturabetaling",
        "fakturabetalning",
        "invoice payment",
    ),
    "qr_code_payments": (
        "qr-code-transaktionen",
        "qr-code transactions",
        "qr-code-zahlungen",
        "qr code transactions",
        "qr-code",
        "qr code",
        "qr-code-betalinger",
        "qr kode betalinger",
        "qr kode-betalinger",
        "qr-code-betaling",
        "qr-kode-betalinger",
        "qr-kode",
        "qr kod",
        "qr-kod",
        "qr kode",
        "codice qr",
        "transazioni con codice qr",
        "código qr",
        "códigos qr",
        "transacciones con códigos qr",
        "transacciones con código qr",
        "kódu qr",
        "kodem qr",
        "pomocí kódu qr",
        "kódom qr",
        "pomocou kódu qr",
        "qr-kooditapahtumat",
        "qr-koodi",
        "qr-kódos",
        "qr-kódos tranzakciók",
    ),
    "donations": (
        "paypal-spendenaktionen",
        "spendenaktionen",
        "spende",
        "donation",
        "donationer",
        "donations",
        "charity donation",
        "don de",
        "donazioni",
    ),
    "nonprofit": (
        "gemeinnützige organisationen",
        "gemeinnützig",
        "gemeinnutzig",
        "nonprofit organisation",
        "nonprofit",
        "non-profit",
        "velgørende",
        "velgørende organisationer",
        "non-profit organisation",
        "organizaciones sin fines de lucro",
        "organizzazioni senza scopo di lucro",
        "organisasjoner",
        "organizacja non-profit",
        "charitativní transakce",
        "charitativních transakcí",
    ),
    "micropayments": (
        "mikrozahlung",
        "micropayment",
        "kleinbetragszahlung",
        "mikrobetaling",
        "mikrobetalinger",
        "mikromaksu",
        "mikropłatność",
        "micropagos",
        "micropaiement",
    ),
    "chargebacks": (
        "rückbuchungsgebühren",
        "rückbuchung",
        "rückabwicklung",
        "chargeback",
    ),
    "disputes": (
        "konfliktgebühren",
        "konfliktgebühr",
        "streitfall",
        "dispute",
    ),
    "refunds": (
        "rückerstattung",
        "rückzahlung",
        "refund",
    ),
    "currency_conversion": (
        "währungsumrechnung",
        "umrechnung",
        "currency conversion",
        "wechselkurs",
    ),
    "withdrawals": (
        "guthaben von einem paypal-geschäftskonto abbuchen",
        "abbuchen",
        "auszahlung",
        "withdrawal",
        "payout",
        "business debit card",
        "business debit",
        "bargeldabhebung",
        "bargeldabbuchung",
        "atm withdrawal",
        "geldautomat",
        "de business debit card",
        "bank account",
        "to a bank account",
        "withdraw to a bank account",
    ),
    "card_verification": (
        "kartenverifizierung",
        "karten verifizierung",
        "card verification",
        "kartenbestätigung",
    ),
    "sepa_direct_debit": (
        "sepa-lastschrift",
        "sepa lastschrift",
        "sepa direct debit",
        "sepa-elv",
        "lastschrift",
        "domiciliación bancaria",
        "addebito diretto",
        "débit direct sepa",
    ),
    "fraud_protection": (
        "professionelles tool zum betrugsschutz",
        "fraud protection advanced",
        "fraud protection",
        "betrugsschutz",
        "outil de protection contre la fraude",
        "protección antifraude",
        "protezione antifrode",
    ),
    "records_request": (
        "anforderung von unterlagen",
        "records request",
        "document request",
        "document fee",
        "dokumentengebühr",
        "demande de documents",
        "solicitud de documentación",
        "richiesta di documenti",
    ),
    "pay_later_consumer": (
        "paypal-ratenzahlungsangebote",
        "ratenzahlungsangebote",
        "pay in 4",
        "pay later",
        "betal senere",
        "afbetaling",
        "paypal pay later",
        "paypal pay later-tilbud",
        "pay later-tilbud",
        "klarna",
        "afterpay",
    ),
    "pos_transactions": (
        "point of sale",
        "paypal point of sale",
        "präsenter karte",
        "card present",
        "kortforevisning",
        "kortterminal",
        "kortterminaler",
        "kortpresent",
    ),
}

# Order used for stable output.
_PRODUCT_ORDER = list(_PRODUCT_ALIASES)

# ---------------------------------------------------------------------------
# Table category detection
# ---------------------------------------------------------------------------

_TABLE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "commercial_rate_table": (
        "standardgebühr beim empfang von inlandstransaktionen",
        "empfangen von inlandstransaktionen",
        "standard transaction fees",
        "commercial transaction fees",
        "geschäftlichen transaktionen",
        "commercial transactions",
        "standardgebyr",
        "standardavgift",
        "standaardtarief",
        "tarifa estándar",
        "tarifa standard",
        "tarifa padrão",
        "tariffa standard",
        "tarification standard",
        "standard taxa",
        "standardtaxa",
        "standardsats",
        "standardowa stawka",
        "standardní sazba",
        "štandardná sadzba",
        "szokásos díja",
        "standard fee",
        "standard rate",
        "modtagelse af indenlandske betalinger",
        "innenlands",
        "indland",
        "innenlandske",
        "inhemska",
        "binnenland",
        "binnenlandse",
        "nationales",
        "transacciones nacionales",
        "nazionali",
        "nacionais",
        "nationella",
        "krajowych",
        "krajinskih",
        "vnitrostátních",
        "domácich",
        "belföldi",
        "εσωτερικές",
        "naționale",
        "национални",
        "nacionalnih",
        "kotimaisten",
        "domestic payments",
        "domestic",
        "commerciële",
        "commerciale",
        "commerciales",
        "kommersielle",
        "kommercielle",
        "kommersiella",
        "kommercielle",
        "kommersielle",
        "komercyjne",
        "komerčních",
        "komerčných",
        "kereskedelmi",
        "εμπορικές",
        "comerciale",
        "comercial",
        "comerciale",
        "tranzakcje",
        "transakcji",
        "transakcí",
        "transakcií",
        "transakció",
        "transakcije",
        "receiving domestic",
        "recepción de transacciones nacionales",
        "recebimento de transações nacionais",
        "ricezione di transazioni nazionali",
        "recepción de transacciones nacionales",
        "ricezione di pagamenti",
    ),
    "online_card_rate_table": (
        "paypal-dienste für online-kartenzahlungen",
        "paypal-dienste für online-zahlungen",
        "online card payments",
        "online-kartenzahlungen",
        "online card",
        "online-kortbetaling",
        "online kort",
        "online kortbetalingstjenester",
        "online-kort",
        "kortbetalingstjenester",
        "kortbetalningstjänster",
        "kortbetalning",
        "kortfinansierad",
        "online kártyás",
        "online kart",
        "online karta",
        "online kortti",
        "online kort",
        "online kartę",
        "online kort",
        "online kart",
        "online kart",
        "online card payment services",
        "online payment services",
        "servizi paypal di pagamento online con carta",
        "servizi di pagamento online con carta",
        "pagamento online con carta",
        "online kartično",
        "online kartica",
        "servizi di pagamento con carta",
        "services de paiement par carte",
        "services de paiement en ligne",
        "service de paiement en ligne",
        "paiement en ligne",
        "serviços de pagamento com cartão",
        "serviços de pagamento online",
        "servicios de pago con tarjeta",
        "servicios de pago en línea",
        "serviços de pagamento online",
        "tarjetas de crédito y débito",
        "kredit- och debitkort",
        "credit and debit card",
        "kredit- og betalingskort",
        "kredit- och betalkort",
        "avancerat kredit- och betalkort",
        "advanced credit and debit card",
        "online-betalningstjänster",
        "online betalingstjenester",
        "online betalingsløsninger",
        "online betalingstjenester",
        "online betaling",
        "online maksut",
        "online maksu",
        "online platby",
        "online platba",
        "služeb paypal pro online platby",
        "služieb paypal pre platby online",
        "online betaalservices van paypal",
        "transacties ontvangen via online betaalservices",
        "pago por internet de paypal",
        "servicios de pago por internet",
        "płatności online kartą w systemie paypal",
        "usług płatności online kartą",
        "online πληρωμές",
        "online πληρωμων",
        "ηλεκτρονικές πληρωμές",
        "ηλεκτρονικες πληρωμες",
        "υπηρεσιών paypal για ηλεκτρονικές",
        "υπηρεσιων paypal για ηλεκτρονικες",
        "ηλεκτρονικές υπηρεσίες",
        "ηλεκτρονικες υπηρεσιες",
    ),
    "goods_and_services_rate_table": (
        "geld für waren und dienstleistungen",
        "goods and services",
    ),
    "donation_rate_table": (
        "empfang von inlandsspenden",
        "donation",
        "spenden",
        "donationer",
        "donationer",
        "donations",
        "donaties",
        "donativos",
        "donativas",
        "doações",
        "donazioni",
        "donationer",
        "donationer",
        "lahjoitukset",
        "darowizn",
        "darowizn krajowych",
        "odbiór darowizn",
        "príspevkov",
        "príspevky",
        "domácich príspevkov",
        "binnenlandse donaties",
        "ontvangen van binnenlandse donaties",
        "adományok",
        "δωρεές",
        "δωρεες",
        "δωρεών",
        "δωρεων",
        "donatii",
        "дарения",
        "donacije",
        "donacijo",
        "donationer",
        "donaties",
        "příspěvky",
        "příspěvků",
        "receiving domestic donations",
        "recepción de donativos nacionales",
        "recebimento de doações domésticas",
        "ricezione di donazioni nazionali",
        "λήψη εγχώριων δωρεών",
        "ληψη εγχωριων δωρεων",
        "sending international donations",
        "receiving international donations",
        "skicka internationella donationer",
        "ta emot internationella donationer",
        "enviar donaciones internacionales",
        "recibir donaciones internacionales",
        "envoyer des dons internationaux",
        "recevoir des dons internationaux",
        "invio di donazioni internazionali",
        "ricezione di donazioni internazionali",
        "wysyłanie międzynarodowych darowizn",
        "odbieranie międzynarodowych darowizn",
        "αποστολή διεθνών δωρεών",
        "λήψη διεθνών δωρεών",
        "senden von internationalen spenden",
        "empfangen von internationalen spenden",
        "lähettää kansainvälisiä lahjoituksia",
        "vastaanottaa kansainvälisiä lahjoituksia",
    ),
    "nonprofit_rate_table": (
        "gemeinnützige organisationen",
        "nonprofit",
        "non-profit",
        "velgørende",
        "velgørende organisationer",
        "non-profit organisation",
        "organizaciones sin fines de lucro",
        "organizaciones benéficas",
        "organizzazioni senza scopo di lucro",
        "enti benefici",
        "organisasjoner",
        "organizacja non-profit",
        "organizacje charytatywne",
        "charitatívnych",
        "charitativních",
        "charitativní transakce",
        "charitativních transakcí",
        "vnitrostátních charitativních transakcí",
        "charitatívnymi príspevkami",
        "domácich transakcií spojených s charitatívnymi príspevkami",
        "jótékonysági",
        "jótékonysági",
        "välgörenhetsorganisationer",
        "välgörenhet",
        "φιλανθρωπικές",
        "φιλανθρωπικά",
        "φιλανθρωπικού",
        "φιλανθρωπικου",
        "instituições de solidariedade",
        "instituição de caridade",
        "institución de solidaridad",
        "entidades sin ánimo de lucro",
        "associazioni di volontariato",
        "caritative",
        "caritat",
        "caridad",
        "caridade",
        "nonguvernamental",
        "humanitar",
        "charity",
        "charitable",
        "liefdadigheid",
        "liefdadigheids",
        "receiving domestic charity",
        "receiving charity",
        "carity transactions",
        "associations caritatives",
    ),
    "apm_rate_table": (
        "alternative zahlungsmethode",
        "alternative payment method",
        "apm-transaktionen",
        "apm",
        "alternativ betalingsmetode",
        "alternative betalingsmetode",
        "alternative betaalmethode",
        "métodos de pago alternativos",
        "metodi di pagamento alternativi",
        "metodo di pagamento alternativo",
        "andere betaalmethode",
        "autre moyen de paiement",
        "autre moyen",
        "autre mode de paiement",
        "alternatív fizetési",
        "alternatívny spôsob platby",
        "alternativní způsob platby",
        "alternatywna forma płatności",
        "alternatywne metody płatności",
        "alternativ betalningsmetod",
        "alternativ betalingsmåte",
        "vaihtoehtoinen maksutapa",
        "vaihtoehtoinen",
        "vmt",
        "apm-transaksjoner",
        "apm-transakcje",
        "apm-transakcje",
        "apm-transakcí",
        "apm-transakcí",
        "apm-transakciók",
        "apm-transakcije",
        "apm-transakcije",
        "alternative betalingsmetoder",
        "alternative payment methods",
        "alternative payment",
    ),
    "pos_rate_table": (
        "point of sale",
        "paypal point of sale",
        "point de vente",
        "points de vente",
        "präsenter karte",
        "kortforevisning",
        "card present",
        "point-of-sale",
        "ponto de venda",
        "punto de venta",
        "punto vendita",
        "punkty sprzedaży",
        "płatności kartą",
        "kassapääte",
        "kassapääte",
        "platební terminál",
        "platební terminály",
        "terminál",
        "kortterminal",
        "kortterminaler",
        "kortläsare",
        "pos-terminal",
        "pos terminal",
        "paypal pos",
        "pos-betalinger",
        "pos transakcije",
        "pos transakcie",
        "pos-transakcije",
        "kassasystem",
        "korttransaktioner",
        "korttransaktion",
        "presentkort",
        "transazioni tramite pos di paypal",
        "transazioni tramite pos",
        "transactions via paypal pos",
        "pos di paypal",
        "tramite pos",
        "via paypal pos",
    ),
    "micropayment_rate_table": (
        "mikrozahlung",
        "micropayment",
        "mikrobetaling",
        "mikrobetalinger",
        "mikromaksu",
        "mikromaksujen",
        "mikrobetaling",
        "mikrobetalinger",
        "mikrobetaling",
        "mikrobetalingar",
        "mikropłatność",
        "mikropłatności",
        "mikrotransakcí",
        "mikrotransakcií",
        "mikrotransakciók",
        "mikrotransakcije",
        "mikrotransakcije",
        "micropagos",
        "micropagos",
        "micropaiement",
        "micropaiements",
        "micropagamentos",
        "mikrobetalning",
        "mikrobetalningar",
        "mikromaksu",
        "μικροπληρωμές",
        "μικροπληρωμες",
        "μικροπληρωμών",
        "μικροπληρωμων",
        "mikroplatby",
        "mikroplatby",
    ),
    "fixed_fee_table": (
        "festgebühr",
        "fixed fee",
        "fast gebyr",
        "fast avgift",
        "tarifa fija",
        "comisión fija",
        "comissão fixa",
        "taxa fixa",
        "vast bedrag",
        "vaste kosten",
        "tariffa fissa",
        "paušální poplatek",
        "opłata stała",
        "frais fixe",
        "commission fixe",
        "kiinteä palkkio",
        "kiinteä maksu",
        "fixný poplatok",
        "fiksna naknada",
        "fiksna provizija",
        "comision fixa",
        "σταθερή χρέωση",
        "ค่าธรรมเนียมคงที่",
        "อัตราคงที่",
        "rögzített díja",
        "rögzített díj",
        "固定費用",
    ),
    "international_surcharge_table": (
        "zusätzliche prozentuale gebühr",
        "prozentuale zusatzgebühr",
        "international surcharge",
        "additional percentage fee",
        "additional percentage-based fee",
        "additional percentage based fee",
        "additional percentage",
        "international payments",
        "internationales",
        "internationella",
        "internationale",
        "internasjonale",
        "kansainvälisten",
        "kansainvälis",
        "medzinárodné",
        "medzinárodné",
        "mezinárodní",
        "zahraniční",
        "nemzetközi",
        "διεθνείς",
        "internaționale",
        "международни",
        "međunarodne",
        "mednarodne",
        "ausland",
        "zusatzgebühr",
        "additional service",
        "service fee",
        "servicegebühr",
        "servicegebyr",
        "serviceavgift",
        "extra percentage",
        "ekstra procentbaseret gebyr",
        "yderligere procentdel af gebyr",
        "extra procentuell avgift",
        "extra procentbaserad avgift",
        "lisäprosenttimaksu",
        "prosenttiperusteinen lisäpalkkio",
        "supplément de commission",
        "sobretasa internacional",
        "tarifa adicional porcentual",
        "tarifa adicional baseada em porcentagem",
        "comissão percentual adicional",
        "tariffa percentuale aggiuntiva",
        "op een percentage gebaseerde",
        "dodatkowa opłata procentowa",
        "dodatečný procentní poplatek",
        "dodatočný percentuálny poplatok",
        "százalékos kiegészítő díja",
        "dodatni postotak",
        "dodatni odstotek",
        "πρόσθετη χρέωση βάσει ποσοστού",
        "προσθετη χρεωση βασει ποσοστου",
        "πρόσθετη χρέωση υπηρεσίας",
        "προσθετη χρεωση υπηρεσιας",
        "διεθνών",
        "διεθνων",
        "διεθνείς",
        "διεθνεις",
        "λήψη διεθνών",
        "ληψη διεθνων",
        "dodatna naknada",
        "dodatna pristojbina",
    ),
    "currency_conversion_table": (
        "währungsumrechnung",
        "umrechnung",
        "currency conversion",
        "guthaben umrechnen",
        "valuutan muuntaminen",
        "valuutanvaihto",
        "valuuttakurssit",
        "valutaomregning",
        "valuuttakurssi",
        "wisselkoers",
        "cambio",
        "cambio de divisa",
        "cambio valuta",
        "conversione",
        "conversão",
        "cambio",
        "devizni",
        "valut",
        "valuuta",
        "valuta",
        "převod",
        "převodu měny",
        "převod zůstatku na firemním účtu",
        "prevod",
        "prevod meny",
        "konverzia",
        "konverzija",
        "μετατροπή",
        "conversie",
        "conversion",
        "valutaveksling",
        "valutakurs",
        "valutakurser",
        "valutakurser",
        "kryptoměna",
        "kryptomena",
        "criptomoeda",
        "criptomoneda",
        "kryptovaluta",
        "converting balance",
        "converting",
    ),
    "other_fees_table": (
        "sonstige gebühren",
        "other fees",
        "autres frais",
        "altre commissioni",
        "otros honorarios",
        "otros cargos",
        "outras taxas",
        "overige kosten",
        "sonstige kosten",
        "overige vergoedingen",
        "diversas taxas",
        "andere kosten",
        "muut kulut",
        "muut maksut",
        "muut kulud",
        "pozostałe opłaty",
        "ostatní poplatky",
        "iné poplatky",
        "ostali troškovi",
        "druge naknade",
        "altres comissions",
        "altres despeses",
        "diger ucretler",
        "diger ücretler",
        "diversi",
        "diverse",
        "miscellaneous fees",
        "additional fees",
    ),
}

# When a caption contains one of these negative signals, it is not treated as a
# rate table even if it also contains product-specific keywords.
_TABLE_NEGATIVE_SIGNALS: dict[str, tuple[str, ...]] = {
    "commercial_rate_table": (
        "spende",
        "donation",
        "donativos",
        "donativas",
        "donaciones",
        "donazione",
        "donazioni",
        "doações",
        "gemeinnützig",
        "nonprofit",
        "non-profit",
        "organizaciones benéficas",
        "organizaciones sin fines de lucro",
        "organizacion benéfica",
        "organizzazioni senza scopo di lucro",
        "organizzazione senza scopo di lucro",
        "enti benefici",
        "ente benefico",
        "organizações beneficentes",
        "organização beneficente",
        "sem fins lucrativos",
        "mikrozahlung",
        "micropayment",
        "alternative zahlungsmethode",
        "alternative payment",
        "online-kartenzahlungen",
        "online card",
        "point of sale",
        "qr-code",
        "qr code",
        "sonstige gebühren",
        "rückbuchung",
        "chargeback",
    ),
    "online_card_rate_table": (
        "sonstige gebühren",
        "rückbuchung",
        "chargeback",
        "betrugsschutz",
        "professionelles tool",
        "professional tool",
    ),
}

# Map a table category to the default schedule name used by its rows.
_TABLE_CATEGORY_SCHEDULE: dict[str, str] = {
    "commercial_rate_table": "commercial",
    "online_card_rate_table": "online_card_payments",
    "goods_and_services_rate_table": "goods_and_services",
    "donation_rate_table": "donations",
    "nonprofit_rate_table": "nonprofit",
    "apm_rate_table": "alternative_payment_methods",
    "pos_rate_table": "pos_transactions",
    "micropayment_rate_table": "micropayments",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=100_000)
def _norm(text: str | None) -> str:
    return clean_text(text or "").lower()


# Pre-normalized product aliases avoid re-normalizing the same static strings on
# every product classification call.
_NORMALIZED_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    product: tuple(_norm(alias) for alias in aliases) for product, aliases in _PRODUCT_ALIASES.items()
}

# Pre-compiled regex for canonical amount validation.
_CANONICAL_AMOUNT_RE = re.compile(r"[+-]?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)")

# Pre-compiled regexes for rate expression parsing.
_PERCENTAGE_RE = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*%")
_PLUS_FIXED_RE = re.compile(r"[+]\s*(.+)")


def _keyword_in_text(text: str, keyword: str) -> bool:
    """Return True when ``keyword`` appears as a whole word/phrase in ``text``."""
    # Use word boundaries to avoid matching the keyword as a substring inside a
    # larger word (e.g. Portuguese "até" inside Czech "přijaté").  This keeps
    # punctuation-delimited tokens such as "<" and ">" working as well.
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return bool(re.search(pattern, text))


def _table_text(table: Table) -> str:
    parts = list(table.section_path or []) + [table.caption or ""]
    for header in table.headers:
        parts.append(header.text)
    return _norm(" ".join(parts))


def _table_context_original(table: Table) -> str:
    """Return original-case table heading context for applicability parsing."""
    parts = list(table.section_path or []) + [table.caption or ""]
    return " ".join(p for p in parts if p)


def _row_cells_text(row: Row) -> list[str]:
    return [c.text for c in row.cells]


def _text_indicates_percentage(text: str | None) -> bool:
    """Return True if text contains a percentage marker or spelling."""
    if not text:
        return False
    lowered = text.lower()
    return "%" in lowered or "prozentpunkt" in lowered or "percentage point" in lowered


def _token_text_indicates_percentage(token) -> bool:
    """Return True if token metadata describes a percentage value.

    PayPal embeds some percentage-point surcharges as raw numbers whose
    internal name or fee-data key contains the word "Prozentpunkte" or
    "percentage points".
    """
    for candidate in (token.raw, token.internal_name, token.fee_data_key):
        if _text_indicates_percentage(candidate):
            return True
    return False


def _first_percentage(row: Row) -> str | None:
    for cell in row.cells:
        cell_indicates_pct = _text_indicates_percentage(cell.text)
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
            if (
                token.kind == "number"
                and token.value
                and (cell_indicates_pct or _token_text_indicates_percentage(token))
            ):
                return token.value
    return None


def _first_money(row: Row) -> tuple[str, str] | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                return token.currency, token.amount
    return None


def _cell_money(cell: Any) -> tuple[str, str] | None:
    for token in cell.tokens:
        if token.kind == "money" and token.amount and token.currency:
            return token.currency, token.amount
    return None


def _row_has_percentage(row: Row) -> bool:
    return _first_percentage(row) is not None


def _row_label(row: Row) -> str:
    """Return the label cell of a row (typically the first non-empty cell)."""
    for cell in row.cells:
        if cell.text.strip():
            return cell.text.strip()
    return ""


def _row_fee_cell(row: Row) -> str:
    """Return the fee/rate cell of a row (typically the last non-empty cell)."""
    for cell in reversed(row.cells):
        if cell.text.strip():
            return cell.text.strip()
    return ""


# Products that may be expressed as a direct fixed monetary amount (no
# percentage and no fixed-fee schedule) in a rate-style row.
_DIRECT_FIXED_FEE_PRODUCTS: frozenset[str] = frozenset(
    {
        "sepa_direct_debit",
        "fraud_protection",
        "records_request",
        "card_verification",
        "withdrawals",
    }
)

# Rate tables whose rows should be scoped to the table's product category.
_CATEGORY_SPECIFIC_TABLES: frozenset[str] = frozenset(
    {
        "nonprofit_rate_table",
        "donation_rate_table",
        "micropayment_rate_table",
        "pos_rate_table",
        "goods_and_services_rate_table",
        "withdrawals_rate_table",
    }
)


# Best-guess default currency for a PayPal market code. Used only when the
# table cells or headers do not carry an explicit ISO 4217 code.
_DEFAULT_CURRENCY_BY_MARKET: dict[str, str] = {
    "DE": "EUR",
    "AT": "EUR",
    "BE": "EUR",
    "NL": "EUR",
    "FR": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "PT": "EUR",
    "IE": "EUR",
    "GR": "EUR",
    "FI": "EUR",
    "LU": "EUR",
    "CY": "EUR",
    "MT": "EUR",
    "SK": "EUR",
    "SI": "EUR",
    "EE": "EUR",
    "LV": "EUR",
    "LT": "EUR",
    "GB": "GBP",
    "US": "USD",
    "CA": "CAD",
    "AU": "AUD",
    "JP": "JPY",
    "CH": "CHF",
    "IN": "INR",
    "BR": "BRL",
    "MX": "MXN",
    "SG": "SGD",
    "HK": "HKD",
    "NZ": "NZD",
    "SE": "SEK",
    "NO": "NOK",
    "DK": "DKK",
    "PL": "PLN",
    "CZ": "CZK",
    "HU": "HUF",
    "IL": "ILS",
    "ZA": "ZAR",
    "AE": "AED",
}


def _infer_currency_for_row(row: Row, table: Table, source: Source | None) -> str | None:
    """Return an explicit or inferred ISO 4217 currency code for a row."""
    # Money tokens already carry a currency.
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.currency:
                return token.currency
    # Look for an explicit three-letter currency code anywhere in the row or headers.
    sources = [cell.text for cell in row.cells]
    sources += [h.text for h in table.headers]
    sources += [table.caption or ""] + list(table.section_path)
    for text in sources:
        for part in re.findall(r"(?<!\w)[A-Z]{3}(?!\w)", text):
            if part in CURRENCY_CODES:
                return part
    # Fall back to the market default from the source URL.
    if source and source.requested_url:
        market = _market_code_from_url(source.requested_url)
        if market:
            return _DEFAULT_CURRENCY_BY_MARKET.get(market)
    return None


def _parse_canonical_amount(amount_str: str) -> str | None:
    """Return a canonical decimal string for an amount that may use thousands or decimal separators."""
    amount_str = amount_str.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if not amount_str:
        return None
    # Reject non-numeric strings before trying to interpret separators.
    if not _CANONICAL_AMOUNT_RE.fullmatch(amount_str):
        return None
    has_dot = "." in amount_str
    has_comma = "," in amount_str
    if not has_dot and not has_comma:
        return normalize_decimal_string(amount_str)
    if has_dot and has_comma:
        last_dot = amount_str.rfind(".")
        last_comma = amount_str.rfind(",")
        if last_dot > last_comma:
            decimal_str = amount_str.replace(",", "")
        else:
            decimal_str = amount_str.replace(".", "").replace(",", ".")
        return normalize_decimal_string(decimal_str)
    sep = "." if has_dot else ","
    parts = amount_str.split(sep)
    if len(parts) == 2:
        int_part, frac_part = parts
        if len(frac_part) in (1, 2):
            decimal = True
        elif len(frac_part) == 3:
            decimal = not (int_part.isdigit() and len(int_part) <= 3)
        else:
            decimal = True
        if decimal:
            if sep == ",":
                return normalize_decimal_string(amount_str.replace(",", "."))
            return normalize_decimal_string(amount_str)
        return int_part + frac_part
    if len(parts[-1]) == 3 and all(len(p) <= 3 for p in parts[:-1]):
        return "".join(parts)
    decimal_str = "".join(parts[:-1]) + "." + parts[-1]
    return normalize_decimal_string(decimal_str)


def _extract_direct_fixed_amounts(
    row: Row,
    product_id: str,
    table: Table,
    source: Source | None,
) -> list[tuple[str, str, str]]:
    """Return direct fixed-fee amounts for a row as (amount, currency, variant_id)."""
    fee_text = _row_fee_cell(row)
    if not fee_text:
        return []
    label = _row_label(row)
    inferred_currency = _infer_currency_for_row(row, table, source)
    amounts: list[tuple[str, str, str]] = []
    # Match numeric amounts with an optional trailing ISO currency code.  The
    # pattern supports both thousands-separated and decimal-comma forms so a
    # value like "50,000.00 IDR" is parsed as a single amount.
    pattern = re.compile(
        r"(?P<operator>[+\-])?(?P<amount>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?P<currency>[A-Za-z]{3})?"
    )
    matches = list(pattern.finditer(fee_text))
    for idx, match in enumerate(matches):
        amount_raw = match.group("amount")
        if not amount_raw:
            continue
        amount = _parse_canonical_amount(amount_raw)
        if amount is None:
            continue
        currency = (match.group("currency") or "").upper() or inferred_currency
        if not currency:
            continue
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(fee_text)
        segment = fee_text[match.start() : next_start]
        variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=segment)
        if variant_id is None:
            variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
        amounts.append((amount, currency, variant_id))
    # Fallback: use the tokenizer's money/number tokens when regex failed to
    # produce a parseable result.
    if not amounts:
        for cell in reversed(row.cells):
            for token in cell.tokens:
                if token.kind == "money" and token.amount and token.currency:
                    amount = _parse_canonical_amount(token.amount)
                    if amount is None:
                        continue
                    variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                    amounts.append((amount, token.currency, variant_id))
                elif token.kind == "number" and token.value and inferred_currency:
                    amount = _parse_canonical_amount(token.value)
                    if amount is None:
                        continue
                    variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                    amounts.append((amount, inferred_currency, variant_id))
    # Explicit zero-fee rows (e.g. "No Fee") for direct fixed-fee products are
    # still fee information and should be represented as a 0 fixed amount.
    if not amounts:
        fee_norm = _norm(fee_text)
        if any(kw in fee_norm for kw in ("no fee", "free", "gratis", "kostenlos", "0.00", "0,00")):
            currency = _infer_currency_for_row(row, table, source)
            if currency:
                variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                amounts.append(("0", currency, variant_id))
    return amounts


_FEE_HEADER_KEYWORDS: frozenset[str] = frozenset(
    {"fee", "gebühr", "charge", "cost", "price", "preis", "betrag", "amount", "tarif", "rate"}
)

_NON_FEE_HEADER_KEYWORDS: frozenset[str] = frozenset(
    {
        "code",
        "vorwahl",
        "prefix",
        "country",
        "market",
        "region",
        "number",
        "no.",
        "phone",
        "telephone",
        "int.",
        "international",
        "markt",
        "land",
    }
)


def _cell_looks_like_fee_cell(header_text: str, table: Table) -> bool:
    """Return True when a numeric cell is plausibly a fee value, not a code/prefix."""
    header_lower = _norm(header_text)
    if any(kw in header_lower for kw in _NON_FEE_HEADER_KEYWORDS):
        return False
    if any(kw in header_lower for kw in _FEE_HEADER_KEYWORDS):
        return True
    table_lower = _norm((table.caption or "") + " ".join(table.section_path))
    return any(kw in table_lower for kw in _FEE_HEADER_KEYWORDS)


def _last_non_empty_cell_text(row: Row) -> str:
    for cell in reversed(row.cells):
        text = cell.text.strip()
        if text:
            return text
    return ""


def _has_likely_numeric_fee_candidate(row: Row, table: Table) -> bool:
    """Return True when a row contains a probable money or number fee value."""
    last_cell_text = _last_non_empty_cell_text(row)
    for i, cell in enumerate(row.cells):
        for token in cell.tokens:
            if token.kind == "money":
                return True
            if token.kind == "number" and token.value:
                header = table.headers[i].text if i < len(table.headers) else ""
                if _cell_looks_like_fee_cell(header, table):
                    return True
                # A bare number in the last (fee) cell of a row is a fee
                # candidate unless the header explicitly marks it as a code or prefix.
                if cell.text.strip() == last_cell_text:
                    header_lower = _norm(header)
                    if not any(kw in header_lower for kw in _NON_FEE_HEADER_KEYWORDS):
                        return True
    return False


# ---------------------------------------------------------------------------
# Product classification
# ---------------------------------------------------------------------------


def _score_label_against_product(label: str, normalized_aliases: tuple[str, ...]) -> int:
    normalized = _norm(label)
    best = 0
    for alias in normalized_aliases:
        if alias == normalized:
            return max(best, len(alias) * 10)
        if alias in normalized:
            best = max(best, len(alias))
    return best


def _classify_product(label: str) -> tuple[str | None, list[str]]:
    """Return the best matching product ID and any ambiguous alternatives."""
    scores: dict[str, int] = {}
    for product_id, aliases in _NORMALIZED_PRODUCT_ALIASES.items():
        score = _score_label_against_product(label, aliases)
        if score:
            scores[product_id] = score
    if not scores:
        return None, []
    max_score = max(scores.values())
    candidates = sorted([pid for pid, sc in scores.items() if sc == max_score])
    if len(candidates) > 1:
        return None, candidates
    return candidates[0], []


# ---------------------------------------------------------------------------
# Table category classification
# ---------------------------------------------------------------------------


# Map a product id to the rate-table category that owns it, used as a fallback
# when the table caption is too generic to classify from metadata alone.
_PRODUCT_CATEGORY_MAP: dict[str, str] = {
    "alternative_payment_methods": "apm_rate_table",
    "advanced_card_payments": "online_card_rate_table",
    "pos_transactions": "pos_rate_table",
    "micropayments": "micropayment_rate_table",
    "donations": "donation_rate_table",
    "nonprofit": "nonprofit_rate_table",
    "goods_and_services": "goods_and_services_rate_table",
    "paypal_checkout": "commercial_rate_table",
    "other_commercial": "commercial_rate_table",
    "guest_checkout": "commercial_rate_table",
    "invoice_pay_later": "commercial_rate_table",
    "pay_later_consumer": "commercial_rate_table",
    "qr_code_payments": "commercial_rate_table",
    "chargebacks": "online_card_rate_table",
    "refunds": "online_card_rate_table",
    "disputes": "online_card_rate_table",
    "card_verification": "apm_rate_table",
    "currency_conversion": "currency_conversion_table",
    "withdrawals": "apm_rate_table",
}


# Default product id for a rate-table category when a row label does not match
# any product alias.  This is used for rows such as "Deutschland" in the
# nonprofit table, where the table context determines the product.
_TABLE_CATEGORY_PRODUCT: dict[str, str] = {
    "nonprofit_rate_table": "nonprofit",
    "donation_rate_table": "donations",
    "micropayment_rate_table": "micropayments",
    "apm_rate_table": "alternative_payment_methods",
    "pos_rate_table": "pos_transactions",
    "online_card_rate_table": "advanced_card_payments",
    "goods_and_services_rate_table": "goods_and_services",
    "commercial_rate_table": "other_commercial",
    "withdrawals_rate_table": "withdrawals",
}


def _is_currency_conversion_text(text: str) -> bool:
    """Return True if the table text describes a currency conversion table."""
    return (
        "währungsumrechnung" in text
        or "umrechnung des guthabens" in text
        or "currency conversion" in text
        or "converting payments" in text
        or "conversions in" in text
        or ("converting" in text and "currency" in text)
        or ("conversion" in text and "currency" in text)
    )


def _is_maximum_fee_table(text: str) -> bool:
    """Return True if the table is a payout/withdrawal maximum fee cap table.

    ``text`` is expected to already be normalized.
    """
    return ("payout" in text or "withdrawal" in text or "withdraw" in text or "payouts" in text) and (
        "maximum fee cap" in text
        or "max fee cap" in text
        or "maximum payout fee" in text
        or "max payout fee" in text
        or ("fee" in text and ("max cap" in text or "maximum cap" in text))
    )


def _is_withdrawals_rate_table(table: Table, text: str) -> bool:
    """Return True if the table is a withdrawals/payouts rate table.

    ``text`` is expected to already be normalized.
    """
    if not (
        "payout" in text
        or "withdrawal" in text
        or "withdraw" in text
        or "payouts" in text
        or "wypłaty" in text
        or "wypłata" in text
        or "výběry" in text
        or "výběr" in text
        or "výbery" in text
    ):
        return False
    # Look for a Rate/% column. Tables that merely list limits or currencies are
    # not rate tables.
    header_text = " ".join(h.text for h in table.headers)
    if "rate" in _norm(header_text) or "%" in header_text:
        return True
    # Some rate tables put the rate in the second column without a header.
    for row in table.rows:
        cells = [c.text for c in row.cells if c.text.strip()]
        if any("%" in c or "rate" in _norm(c) for c in cells):
            return True
    return False


def _table_has_fixed_fee_rate(table: Table) -> bool:
    """Return True when any data row of ``table`` contains a percentage plus a fixed fee."""
    for row in table.rows:
        fee_text = _row_fee_cell(row)
        _, has_fixed = _parse_rate_expression(fee_text)
        if has_fixed:
            return True
    return False


_FIXED_FEE_KEYWORDS = (
    "festgebühr",
    "fixed fee",
    "fast gebyr",
    "fast avgift",
    "tarifa fija",
    "comisión fija",
    "comissão fixa",
    "tarifa fixa",
    "taxa fixa",
    "vast bedrag",
    "vaste kosten",
    "tariffa fissa",
    "paušální poplatek",
    "pevný poplatek",
    "opłata stała",
    "frais fixe",
    "commission fixe",
    "kiinteä palkkio",
    "kiinteä maksu",
    "fixný poplatok",
    "pevný poplatok",
    "fiksna naknada",
    "fiksna provizija",
    "comision fixa",
    "σταθερή χρέωση",
    "ค่าธรรมเนียมคงที่",
    "อัตราคงที่",
    "rögzített díja",
    "rögzített díj",
    "fix díj",
    "固定費用",
)


_MIN_MAX_FEE_KEYWORDS = (
    "mindest",
    "höchst",
    "minimum",
    "maximum",
    "minim",
    "maxim",
    "mínim",
    "máxim",
    "massim",
    "minsta",
    "största",
    "minste",
    "maks",
    "lavest",
    "højest",
    "obergrenze",
    "obergr",
    "limit",
    "cap",
    "ceiling",
    "floor",
    "payout maximum",
    "withdrawal limit",
    "transaction limit",
    "send limit",
    "receive limit",
    "mindestbetrag",
    "höchstbetrag",
    "max cap",
    "max limit",
    "maximum cap",
)


_DIRECT_FIXED_FEE_KEYWORDS = (
    "chargeback",
    "rückbuchung",
    "rückbuchungs",
    "dispute",
    "streit",
    "claim",
    "withdrawal",
    "auszahlung",
    "auszahlungen",
    "payout",
    "verification",
    "verifizierung",
    "authorization",
    "autorisierung",
    "refund",
    "rückerstattung",
    "rückerstattungen",
    "erstattung",
    "terugbetaling",
)


_INTERNATIONAL_SURCHARGE_KEYWORDS = (
    "prozentuale zusatzgebühr",
    "zusätzliche prozentuale gebühr",
    "international surcharge",
    "receiving international transactions",
    "sending international transactions",
    "additional percentage fee",
    "additional percentage-based fee",
    "additional percentage based fee",
    "additional percentage",
    "international payments",
    "ekstra procentbaseret gebyr",
    "yderligere procentdel af gebyr",
    "extra procentuell avgift",
    "extra procentbaserad avgift",
    "lisäprosenttimaksu",
    "prosenttiperusteinen lisäpalkkio",
    "supplément de commission",
    "sobretasa internacional",
    "extra percentage",
    "service fee",
    "servicegebühr",
    "servicegebyr",
    "serviceavgift",
    "service charge",
    "servicekosten",
    "servicekostnad",
    "servicekostnader",
    "yderligere servicegebyr",
    "ytterligare serviceavgift",
    "tarifa de servicio adicional",
    "comissão de serviço adicional",
    "tariffa di servizio aggiuntiva",
    "frais de service supplémentaires",
    "dodatkowa opłata za usługę",
    "dodatečný poplatek za služby",
    "dodatočný poplatok za služby",
    "πρόσθετη χρέωση υπηρεσίας",
    "további szolgáltatási díj",
    "tarifa adicional porcentual",
    "comisión porcentual adicional",
    "comisión adicional con base en un porcentaje",
    "tarifa adicional baseada em porcentagem",
    "comissão percentual adicional",
    "tariffa percentuale aggiuntiva",
    "op een percentage gebaseerde",
    "dodatkowa opłata procentowa",
    "dodatečný procentní poplatek",
    "dodatočný percentuálny poplatok",
    "százalékos kiegészítő díja",
    "dodatni postotak",
    "dodatni odstotek",
    "πρόσθετη χρέωση βάσει ποσοστού",
    "additional service",
    "additional service fee",
)


def _classify_table_category(table: Table) -> str | None:
    text = _table_text(table)
    # Explicit schedule-type captions are authoritative and win over product
    # rate-table keywords such as "commercial transactions" or "donations".
    fixed_fee_keywords = _FIXED_FEE_KEYWORDS
    if any(kw in text for kw in fixed_fee_keywords):
        return "fixed_fee_table"

    # Maximum fee cap tables (e.g. "Maximum fee cap for PayPal Payouts") are
    # fee schedules, not generic limits.
    if _is_maximum_fee_table(text):
        return "maximum_fee_table"

    # Limits, caps, min/max and ceiling/floor tables are not transaction fees
    # and must be detected before direct fixed or rate-table keywords.
    min_max_fee_keywords = _MIN_MAX_FEE_KEYWORDS
    if any(kw in text for kw in min_max_fee_keywords):
        return "min_max_fee_table"

    # Some tables are captioned with international-surcharge language but
    # actually list full transaction rates (percentage + fixed fee) by buyer
    # country. Classify these as commercial rate tables so the rows become
    # product rules with market applicability instead of surcharge schedules.
    if (
        "receiving international transactions" in text or "sending international transactions" in text
    ) and _table_has_fixed_fee_rate(table):
        return "commercial_rate_table"

    # Withdrawals/payouts with a Rate column are rate tables (e.g. "Sending
    # PayPal Payouts"). This must come before direct_fixed because those
    # keywords also match "payout" / "withdrawal".
    if _is_withdrawals_rate_table(table, text):
        return "withdrawals_rate_table"

    # Direct monetary fee tables (chargebacks, disputes, withdrawals, refunds,
    # card verification, authorisation) are not generic fixed-fee schedules and
    # must be identified separately.
    direct_fixed_fee_keywords = _DIRECT_FIXED_FEE_KEYWORDS
    if any(kw in text for kw in direct_fixed_fee_keywords):
        return "fixed_fee_table"

    international_surcharge_keywords = _INTERNATIONAL_SURCHARGE_KEYWORDS
    if any(kw in text for kw in international_surcharge_keywords):
        return "international_surcharge_table"
    if _is_currency_conversion_text(text):
        return "currency_conversion_table"

    category = _select_category_from_scores(table, text)
    # Tables that score as international surcharge schedules but actually
    # contain full percentage + fixed-fee rates are commercial rate tables
    # (e.g. "Receiving international transactions").
    if category == "international_surcharge_table" and _table_has_fixed_fee_rate(table):
        return "commercial_rate_table"
    return category


_LIMIT_OR_CAP_KEYWORDS = (
    "withdrawal limit",
    "withdrawal limits",
    "payout limit",
    "payout limits",
    "payout maximum",
    "payout maximums",
    "payout max",
    "payout minimum",
    "payout min",
    "transaction limit",
    "transaction limits",
    "send limit",
    "send limits",
    "receive limit",
    "receive limits",
    "mindest",
    "höchst",
    "minimum",
    "maximum",
    "minim",
    "maxim",
    "mínim",
    "máxim",
    "massim",
    "minsta",
    "största",
    "minste",
    "maks",
    "lavest",
    "højest",
    "obergrenze",
    "obergr",
    "untergrenze",
    "untergr",
    "limit",
    "limit cap",
    "cap",
    "ceiling",
    "floor",
    "max cap",
    "max limit",
    "maximum cap",
    "min cap",
    "minimum cap",
    "mindestbetrag",
    "höchstbetrag",
    "no more than",
    "not more than",
    "not less than",
    "at least",
    "at most",
    "up to",
    "up-to",
)


def _is_limit_or_cap_row(label: str, fee_text: str = "") -> bool:
    """Return True if a row describes a limit, cap, min/max or ceiling/floor."""
    if _text_indicates_percentage(fee_text):
        return False
    combined = _norm(label + " " + fee_text)
    return any(kw in combined for kw in _LIMIT_OR_CAP_KEYWORDS)


def _select_category_from_scores(table: Table, text: str) -> str | None:
    """Score table text against category keywords and resolve the best category."""
    scores: dict[str, int] = {}
    for category, keywords in _TABLE_CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_norm = _norm(kw)
            if kw_norm in text:
                score = max(score, len(kw_norm))
        if score:
            scores[category] = score
    if not scores:
        return _classify_table_by_row_labels(table)
    candidates = _top_category_candidates(scores)
    candidates = _filter_category_negative_signals(candidates, text)
    if not candidates:
        return _fallback_category_candidate(scores, text, table)
    return candidates[0]


def _top_category_candidates(scores: dict[str, int]) -> list[str]:
    max_score = max(scores.values())
    return [cat for cat, sc in scores.items() if sc == max_score]


def _filter_category_negative_signals(candidates: list[str], text: str) -> list[str]:
    kept: list[str] = []
    for category in candidates:
        negatives = _TABLE_NEGATIVE_SIGNALS.get(category, ())
        if any(_norm(neg) in text for neg in negatives):
            continue
        kept.append(category)
    return kept


def _fallback_category_candidate(scores: dict[str, int], text: str, table: Table) -> str | None:
    # If the top candidates were removed, fall back to the next-highest-scoring
    # category or to row-label inference.
    removed = set(_TABLE_NEGATIVE_SIGNALS.keys())
    remaining = {cat: sc for cat, sc in scores.items() if cat not in removed}
    if not remaining:
        remaining = scores
    next_score = max(remaining.values())
    candidates = [cat for cat, sc in remaining.items() if sc == next_score]
    candidates = _filter_category_negative_signals(candidates, text)
    if len(candidates) == 1:
        return candidates[0]
    return _classify_table_by_row_labels(table)


def _classify_table_by_row_labels(table: Table) -> str | None:
    """Infer a table category from the product ids of its rows."""
    category_counts: dict[str, int] = {}
    for row in table.rows:
        label = _row_label(row)
        if not label:
            continue
        product_id, _ = _classify_product(label)
        if product_id:
            category = _PRODUCT_CATEGORY_MAP.get(product_id)
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
    if not category_counts:
        return None
    best_category, _ = max(category_counts.items(), key=lambda kv: kv[1])
    # Negative signals can override a row-label inference (e.g. "Other fees"
    # tables that happen to contain a credit-card row should be ignored).
    negatives = _TABLE_NEGATIVE_SIGNALS.get(best_category, ())
    text = _table_text(table)
    if any(_norm(neg) in text for neg in negatives):
        return None
    return best_category


# ---------------------------------------------------------------------------
# Schedule naming
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# APM method extraction
# ---------------------------------------------------------------------------

_APM_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "thai_online_bank_transfer": (
        "thai online bank transfer",
        "thailändsk online bank transfer",
        "thailandsk online bankoverførsel",
        "thailändische online banküberweisung",
        "thailändische online-banküberweisung",
        "thailändische online bank überweisung",
        "thai online bank",
        "thai online bankoverførsel",
        "thai online banküberweisung",
        "thai online bankoverschrijving",
        "thai online bankovní převod",
        "thai online bankový prevod",
        "thai online banki átutalás",
        "thai virement bancaire en ligne",
        "thai bonifico bancario online",
        "thai transferencia bancaria online",
        "thai transferência bancária online",
        "thai verkkopankkisiirto",
        "thailändsk",
        "thailandsk",
        "thaimaan",
        "thaimaa",
        "thaiföldi",
        "thaïlandaise",
        "tailandesa",
        "tailandese",
        "tailandés",
        "tajlandzkiej",
        "thajsku",
        "thajska",
        "thajském",
        "thajskom",
        "thai",
    ),
    "latvian_online_bank_transfer": (
        "latvian online bank transfer",
        "lettisk online bank transfer",
        "latvijas online bank transfer",
        "latvian online bankoverførsel",
        "lettische online banküberweisung",
        "lettische online-banküberweisung",
        "lettische online bank überweisung",
        "latvian online bank",
        "latvian online banküberweisung",
        "latvian online bankoverschrijving",
        "latvian online bankovní převod",
        "latvian online bankový prevod",
        "latvian online banki átutalás",
        "latvian virement bancaire en ligne",
        "latvian bonifico bancario online",
        "latvian transferencia bancaria online",
        "latvian transferência bancária online",
        "latvian verkkopankkisiirto",
        "lettisk",
        "lettische",
        "latvijas",
        "latvijska",
        "läti",
        "lettországi",
        "letonska",
        "letonské",
        "letonskom",
        "latvian",
    ),
    "lithuanian_online_bank_transfer": (
        "lithuanian online bank transfer",
        "litauisk online bank transfer",
        "lietuvos online bank transfer",
        "lithuanian online bankoverførsel",
        "litauische online banküberweisung",
        "litauische online-banküberweisung",
        "litauische online bank überweisung",
        "lithuanian online bank",
        "lithuanian online banküberweisung",
        "lithuanian online bankoverschrijving",
        "lithuanian online bankovní převod",
        "lithuanian online bankový prevod",
        "lithuanian online banki átutalás",
        "lithuanian virement bancaire en ligne",
        "lithuanian bonifico bancario online",
        "lithuanian transferencia bancaria online",
        "lithuanian transferência bancária online",
        "lithuanian verkkopankkisiirto",
        "litauisk",
        "litauische",
        "lie tuvos",
        "liettualainen",
        "lietuvos",
        "litván",
        "litewska",
        "litewskim",
        "lithuanian",
    ),
    "estonian_online_bank_transfer": (
        "estonian online bank transfer",
        "estonian online bank",
        "estonia online bank transfer",
        "estonia online bank",
        "estonia banks",
        "estonian banks",
        "estonian bank transfer",
        "estnische online banküberweisung",
        "estnische online bank",
        "online bankovní převod v estonsku",
        "online bankový prevod v estónsku",
        "online banki átutalás észtország",
        "virement bancaire en ligne estonien",
        "bonifico bancario online estone",
        "transferencia bancaria online de estonia",
        "transferência bancária online da estónia",
        "estnisk",
        "estnische",
        "esti",
        "estonia",
        "estonian",
        "estonsku",
        "estónsku",
        "estnischen",
    ),
    "online_bank_transfer": (
        "online bank transfer",
        "online bankoverførsel",
        "online banküberweisung",
        "online-banküberweisung",
        "online bank überweisung",
        "online bankoverschrijving",
        "online bankovní převod",
        "online bankový prevod",
        "online banki átutalás",
        "virement bancaire en ligne",
        "bonifico bancario online",
        "transferencia bancaria online",
        "transferência bancária online",
        "verkkopankkisiirto",
        "bank transfer",
        "banktransfer",
        "online bank",
        "banküberweisung",
        "bankoverschrijving",
        "banköverföring",
        "bankoverførsel",
    ),
    "skrill": ("skrill",),
    "ovo_premium": (
        "ovo premium",
        "ovo",
    ),
    "gopay": (
        "gopay",
        "go pay",
    ),
    "blik_pay_later": (
        "blik pay later",
        "blik",
    ),
    "kredivo": ("kredivo",),
    "floa_pay": (
        "floa pay",
        "floa",
    ),
    "scalapay": ("scalapay",),
    "naver_pay": (
        "naver pay",
        "naverpay",
    ),
    "picpay": ("picpay",),
    "nupay": ("nupay",),
    "true_money": (
        "true money",
        "truemoney",
    ),
    "airtel": ("airtel",),
    "pago_efectivo": (
        "pago efectivo",
        "pagoefectivo",
    ),
    "mercado_pago": (
        "mercado pago",
        "mercadopago",
    ),
    "pesa": (
        "pesa",
        "m-pesa",
        "mpesa",
    ),
    "shopee_pay": (
        "shopee pay",
        "shopeepay",
    ),
}

_APM_SPECIAL_METHOD_IDS: frozenset[str] = frozenset(
    [
        "thai_online_bank_transfer",
        "latvian_online_bank_transfer",
        "lithuanian_online_bank_transfer",
        "estonian_online_bank_transfer",
        "online_bank_transfer",
        "skrill",
        "ovo_premium",
        "gopay",
        "blik_pay_later",
        "kredivo",
        "floa_pay",
        "scalapay",
        "naver_pay",
        "picpay",
        "nupay",
        "true_money",
        "airtel",
        "pago_efectivo",
        "mercado_pago",
        "pesa",
        "shopee_pay",
        "twint",
        "doku_wallet",
        "linkaja",
        "jenius_pay",
        "paysera",
        "dragonpay",
        "codi",
        "halopesa",
        "mixx_by_yas",
        "payattitude",
        "pesalink",
        "promptpay_qr",
        "pse",
        "bre_b",
        "nibss_bank_transfer",
        "nequi",
        "vietqr",
        "coins",
        "paysafecard",
        "oxxopay",
        "klarna",
        "fiuu_cash",
        "opay",
        "apple_pay",
        "wire_transfer",
        "spei",
        "swish",
        "dimo",
    ]
)

# Sort aliases by length descending so the longest/most specific phrase wins
# (e.g. "thai online bank transfer" before "online bank transfer").
_APM_SORTED_ALIASES: list[tuple[str, str]] = sorted(
    [(canonical, alias) for canonical, aliases in _APM_METHOD_ALIASES.items() for alias in aliases],
    key=lambda item: (-len(item[1]), item[0], item[1]),
)


_APM_SEPARATOR_RE = re.compile(
    r"[,;/]|"
    r"(?:\s+(?:and|und|i|y|et|og|ja|oraz|och|e)\s+)",
    re.IGNORECASE,
)
_APM_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_APM_EXAMPLE_PHRASE_RE = re.compile(r"\b(z\s*\.\s*b\s*\.?|e\s*\.\s*g\s*\.?|np\.|ex\.?)\b")

# Full phrases that indicate a label part is a generic APM header, not a method.
_APM_HEADER_PHRASES: set[str] = {
    "alle anderen alternativen zahlungsmethoden",
    "alle anderen alternativen zahlungsmethode",
    "alternative zahlungsmethode",
    "alternative zahlungsmethoden",
    "alternative payment method",
    "alternative payment methods",
    "alternative payment methods (apms) through pay links and buttons",
    "all other alternative payment methods",
    "all other alternative payment method",
    "all other apm",
    "standard credit and debit card payments",
    "credit and debit card payments",
    "other third-party wallets",
    "third-party wallets",
    "third party digital wallet",
    "third-party digital wallet",
    "autres moyens de paiement alternatifs",
    "autres moyens de paiement",
    "altre modalità di pagamento alternative",
    "otros métodos de pago alternativos",
    "otras carteras externas",
    "u otras carteras externas",
    "andere betaalmethode",
    "andere betaalmethoden",
    "andere alternatieve betaalmethoden",
    "andere alternative betalingsmetoder",
    "andere alternative zahlungsmethoden",
    "andere alternative zahlungsmethode",
    "andre alternative betalingsmetoder",
    "pozostałe metody płatności",
    "inne płatności z użyciem alternatywnych form płatności",
    "wszystkie inne alternatywne formy płatności",
    "άλλες εναλλακτικές μέθοδοι πληρωμής",
    "εναλλακτικές μέθοδοι πληρωμής",
    "övriga alternativa betalningsmetoder",
    "muut vaihtoehtoiset maksutavat",
    "más alternatív fizetési módok",
    "outros métodos de pagamento alternativos",
    "alte metode alternative de plată",
    "ostatné alternatívne spôsoby platby",
    " Ostali alternativni načini plačila",
    "alternatív fizetési módok",
    "alternative betalingsmåter",
    "alternative betalingsmetoder",
    "alternative betalingsmåder",
    "alternative betalningsmetoder",
    "alternative maksutavat",
    "alternativna sredstva plaćanja",
    "alternativni načini plaćanja",
    "alternativne metode plaćanja",
    "alternatívne spôsoby platby",
    "alternativní způsoby platby",
    "alternativní platební metody",
    "alternativne metody płatności",
    "alternatyvūs mokėjimo būdai",
    "alternatīvie maksājumi",
    "alternatīvie maksājumu veidi",
    "alternatívne platobné metódy",
    "alternativne metode plačila",
    "alternativne plačilne metode",
    "alternativni plačilni sistemi",
    "alternativni načini plačevanja",
    "alternativna plačilna sredstva",
    "alternativni plačilni mehanizmi",
    "alternativne metode plačevanja",
    "alternativne finančne storitve",
    "alternativne plačilne rešitve",
    "apm-transaktionsgebühren",
    "apm-transaktion",
    "apm-transaksjoner",
    "apm-transakcje",
    "apm-transakcija",
    "apm transactions",
    "apm transaction",
    "apm-transakciók",
    "apm-maksut",
    "apm-maksutapa",
    "apm-betalinger",
    "apm-betaling",
    "apm-betalning",
    "apm-betalningar",
    "apm-zahlungen",
    "apm-zahlung",
    "apm-maks",
    "apm",
    "abm",
    "vmt",
    "ofertas de pay later de paypal",
    "tarjeta de crédito",
    "tarjeta de débito",
    "buy buttons",
    "shopping cart buttons",
    "payment links",
    "wire transfer",
    "transfer to debit card",
    "cash a check",
    "standard credit",
    "debit card payments",
    "credit card payments",
    "standard debit",
}

# Tokens that indicate a label part is a generic APM header, not a method list.
_APM_HEADER_TOKENS: set[str] = {
    "alternative",
    "zahlungsmethode",
    "zahlungsmethoden",
    "payment",
    "payments",
    "method",
    "methods",
    "método",
    "métodos",
    "metodo",
    "metodi",
    "betaalmethode",
    "betaalmethoden",
    "moyen",
    "paiement",
    "pagamento",
    "pagamenti",
    "płatności",
    "maksutapa",
    "maksutavat",
    "platba",
    "mód",
    "módy",
    "apm",
    "apms",
    "abm",
    "vmt",
    "debito",
    "débito",
    "credito",
    "crédito",
    "debit",
    "credit",
    "card",
    "cards",
    "standard",
}

# Token sets for the individual payment methods we can extract from APM labels.
_THAI_TOKENS = {
    "thai",
    "thailand",
    "thailändisch",
    "thailändische",
    "thailändsk",
    "thailandsk",
    "thaimaan",
    "thaimaa",
    "thaiföldi",
    "thaïlandaise",
    "tailandesa",
    "tailandese",
    "tailandés",
    "tajlandzkiej",
    "thajsku",
    "thajska",
    "thajském",
    "thajskom",
    "tailandia",
    "tailândia",
    "thailande",
    "thaïlande",
}
_LATVIAN_TOKENS = {
    "latvian",
    "lettisch",
    "lettische",
    "latvijas",
    "latvijska",
    "läti",
    "lettországi",
    "letonska",
    "letonské",
    "letonskom",
    "letonia",
    "letonie",
    "letonië",
    "lettland",
    "letland",
}
_LITHUANIAN_TOKENS = {
    "lithuanian",
    "litauisch",
    "litauische",
    "liettualainen",
    "lietuvos",
    "litván",
    "litewska",
    "litewskim",
    "lie",
    "tuvos",
    "lituania",
    "lituanie",
    "litouwen",
    "litauen",
}
_ESTONIAN_TOKENS = {
    "estonian",
    "estnische",
    "estnisk",
    "estniska",
    "estlandsk",
    "esztorszagi",
    "esti",
    "estonia",
    "estonsku",
    "estónsku",
    "estnischen",
}
_BANK_TOKENS = {
    "bank",
    "banküberweisung",
    "bankuberweisung",
    "banktransfer",
    "bankoverførsel",
    "bankoverschrijving",
    "bankovní",
    "bankový",
    "banki",
    "átutalás",
    "virement",
    "bonifico",
    "transferencia",
    "transferência",
    "bancaire",
    "bancaria",
    "banco",
    "bancos",
    "banks",
    "banköverföring",
    "verkkopankki",
    "verkkopankkisiirto",
    "pankki",
    "siirto",
    "überweisung",
    "uberweisung",
    "transfer",
    "transfert",
    "overschrijving",
    "overførsel",
    "prevod",
    "przelew",
    "platba",
    "bankkonto",
    "bankkonten",
    "account",
}
_ONLINE_TOKENS = {
    "online",
    "on-line",
    "on",
    "line",
    "ligne",
    "línea",
    "elektronikus",
    "elektronische",
    "elektronisch",
    "eletrônico",
    "electronico",
    "elettronico",
    "internet",
    "verkkopankki",
}

_APM_METHOD_MATCHERS: list[tuple[str, list[set[str]], set[str]]] = [
    ("thai_online_bank_transfer", [_THAI_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("latvian_online_bank_transfer", [_LATVIAN_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("lithuanian_online_bank_transfer", [_LITHUANIAN_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("estonian_online_bank_transfer", [_ESTONIAN_TOKENS, _BANK_TOKENS], set()),
    (
        "online_bank_transfer",
        [_ONLINE_TOKENS, _BANK_TOKENS],
        _THAI_TOKENS | _LATVIAN_TOKENS | _LITHUANIAN_TOKENS | _ESTONIAN_TOKENS,
    ),
    ("skrill", [{"skrill"}], set()),
    ("ovo_premium", [{"ovopremium", "ovo", "ovos"}], set()),
    ("gopay", [{"gopay", "go", "gopa"}], set()),
    ("blik_pay_later", [{"blikpaylater", "blik"}], set()),
    ("kredivo", [{"kredivo"}], set()),
    ("floa_pay", [{"floapay", "floa"}], set()),
    ("scalapay", [{"scalapay"}], set()),
    ("naver_pay", [{"naverpay", "naver"}], set()),
    ("picpay", [{"picpay"}], set()),
    ("nupay", [{"nupay"}], set()),
    ("true_money", [{"truemoney", "true"}], set()),
    ("airtel", [{"airtel"}], set()),
    ("pago_efectivo", [{"pagoefectivo", "pago"}], set()),
    ("mercado_pago", [{"mercadopago", "mercado"}], set()),
    ("pesa", [{"pesa", "mpesa"}], set()),
    ("shopee_pay", [{"shopeepay", "shopee"}], set()),
    ("twint", [{"twint"}], set()),
    ("doku_wallet", [{"dokuwallet", "doku"}], set()),
    ("linkaja", [{"linkaja"}], set()),
    ("jenius_pay", [{"jeniuspay"}], set()),
    ("paysera", [{"paysera"}], set()),
    ("dragonpay", [{"dragonpay"}], set()),
    ("codi", [{"codi"}], set()),
    ("halopesa", [{"halopesa"}], set()),
    ("mixx_by_yas", [{"mixxbyyas"}], set()),
    ("payattitude", [{"payattitude"}], set()),
    ("pesalink", [{"pesalink"}], set()),
    ("promptpay_qr", [{"promptpayqr"}], set()),
    ("pse", [{"pse"}], set()),
    ("bre_b", [{"breb"}], set()),
    ("nibss_bank_transfer", [{"nibbs", "nigerian", "nibss"}], set()),
    ("nequi", [{"nequi"}], set()),
    ("vietqr", [{"vietqr"}], set()),
    ("coins", [{"coins"}], set()),
    ("paysafecard", [{"paysafecard"}], set()),
    ("oxxopay", [{"oxxopay"}], set()),
    ("klarna", [{"klarna"}], set()),
    ("venmo", [{"venmo"}], set()),
    ("fiuu_cash", [{"fiuucash"}], set()),
    ("opay", [{"opay"}], set()),
    ("apple_pay", [{"applepay"}], set()),
    ("wire_transfer", [{"wire"}, {"transfer"}], set()),
    ("spei", [{"spei"}], set()),
    ("swish", [{"swish"}], set()),
    ("dimo", [{"dimo"}], set()),
]


def _tokenize_apm_label(part_norm: str) -> set[str]:
    """Tokenize an APM label part for robust method matching.

    Collapses multi-word method names (e.g. "go pay", "ovo premium") into a
    single token so they can be matched with word boundaries.
    """
    # Normalize punctuation to spaces, then pre-join multi-word brand names.
    joined = _APM_PUNCTUATION_RE.sub(" ", part_norm)
    joined = (
        joined.replace("go pay", "gopay")
        .replace("ovo premium", "ovopremium")
        .replace("floa pay", "floapay")
        .replace("blik pay later", "blikpaylater")
        .replace("naver pay", "naverpay")
        .replace("true money", "truemoney")
        .replace("pago efectivo", "pagoefectivo")
        .replace("mercado pago", "mercadopago")
        .replace("shopee pay", "shopeepay")
        .replace("m pesa", "mpesa")
        .replace("doku wallet", "dokuwallet")
        .replace("jenius pay", "jeniuspay")
        .replace("mixx by yas", "mixxbyyas")
        .replace("pay attitude", "payattitude")
        .replace("promptpay qr", "promptpayqr")
        .replace("bre b", "breb")
        .replace("viet qr", "vietqr")
        .replace("pay safe card", "paysafecard")
        .replace("paysafe card", "paysafecard")
        .replace("oxxo pay", "oxxopay")
        .replace("fiuu cash", "fiuucash")
        .replace("o pay", "opay")
        .replace("air tel", "airtel")
        .replace("kre divo", "kredivo")
        .replace("bankoverschrij ving", "bankoverschrijving")
        .replace("apple pay", "applepay")
    )
    return set(joined.split())


def _extract_apm_methods(label: str) -> tuple[list[str], list[str]]:
    """Extract canonical payment-method IDs from an APM row label.

    Returns (canonical_ids, unknown_segments). The canonical IDs are sorted and
    deduplicated. Unknown segments are raw label parts that did not match any
    known method. Token-based matching avoids false positives like "Republik"
    containing "blik" or "Thailändische Baht" containing "thai".
    """
    norm = _norm(label)
    if not norm:
        return [], []

    parts = [p.strip() for p in _APM_SEPARATOR_RE.split(label) if p.strip()]
    if not parts:
        parts = [label]

    methods: set[str] = set()
    unknowns: list[str] = []

    for part in parts:
        part_norm = _norm(part)
        if not part_norm or len(part_norm) < 3:
            continue

        # Drop introductory phrases like "e.g." or "z.b.".
        part_norm = _APM_EXAMPLE_PHRASE_RE.sub("", part_norm).strip()
        if not part_norm:
            continue

        # Skip generic header phrases ("Alternative payment method", "Alle anderen...").
        if any(phrase in part_norm for phrase in _APM_HEADER_PHRASES):
            continue

        tokens = _tokenize_apm_label(part_norm)

        # Skip parts that are only header tokens.
        if tokens & _APM_HEADER_TOKENS and not (tokens - _APM_HEADER_TOKENS):
            continue

        matched: str | None = None
        for method_id, required_groups, forbidden in _APM_METHOD_MATCHERS:
            if tokens & forbidden:
                continue
            if all(tokens & group for group in required_groups):
                matched = method_id
                break

        if matched:
            methods.add(matched)
        else:
            unknowns.append(part)

    return sorted(methods), sorted(set(unknowns))


def _is_apm_special_label(label: str) -> bool:
    """Return True if a row label describes APM special methods.

    These labels list multiple alternative payment methods (e.g. Thai online
    bank transfer, Skrill, BLIK, Kredivo, etc.) and would otherwise be
    misclassified because they contain substrings like "pay later" (from
    "BLIK Pay Later") that collide with invoice_pay_later / pay_later_consumer.
    """
    methods, _ = _extract_apm_methods(label)
    if not methods:
        return False
    # A spurious "online_bank_transfer" match can be triggered by generic tokens
    # such as "on" + "bank" in a withdrawal/return row (e.g. "Bank Return on
    # Withdrawal/Transfer out of PayPal"). Do not treat those as APM special.
    if methods == ["online_bank_transfer"]:
        text = _norm(label)
        if any(kw in text for kw in ("withdrawal", "return", "chargeback", "refund")):
            return False
    return any(m in _APM_SPECIAL_METHOD_IDS for m in methods)


def _is_international_label(label: str) -> bool:
    """Return True if the label describes an international/cross-border fee."""
    text = _norm(label)
    return any(
        kw in text
        for kw in (
            "internationaux",
            "internacionais",
            "internacional",
            "internazionali",
            "foreign",
            "outside",
            "ausland",
            "ausländ",
            "außerhalb",
            "non-eea",
            "non eea",
            "non-eu",
            "non eu",
            "fuera de",
            "fuera",
            "hors",
            "estrangeiro",
            "estrangeira",
            "utenlands",
            "utland",
            "ulkomaan",
            "interna",
            "külföld",
            "külföldi",
            "međunarod",
            "međunarodne",
            "zahraniční",
            "zahraničné",
            "zagraniczny",
            "zagraniczne",
            "zagraniczna",
            "zagranicznych",
            "międzynarodowe",
            "międzynarodowej",
            "międzynarodowych",
            "międzynarodowa",
            "międzynarodowy",
            "mednarodne",
            "mednarodni",
            "kansainvälinen",
            "kansainvalinen",
            "διεθνείς",
            "διεθνεις",
        )
    )


def _is_domestic_label(label: str) -> bool:
    """Return True if the label describes a domestic/in-country fee."""
    text = _norm(label)
    if "international" in text:
        return False
    return any(
        kw in text
        for kw in (
            "domestic",
            "doméstico",
            "domésticos",
            "doméstica",
            "domésticas",
            "domesticos",
            "domesticas",
            "domestici",
            "domestic",
            "inland",
            "innenland",
            "innenlands",
            "national",
            "nacional",
            "nacionais",
            "national",
            "local",
            "lokal",
            "lokal",
            "nasional",
            "inland",
            "innenlands",
            "inländer",
            "inlander",
            "krajowe",
            "krajowa",
            "krajowych",
            "krajowy",
            "krajových",
            "domácich",
            "domestic",
            "εγχώριες",
            "εγχωριες",
        )
    )


def _first_variant_match(text: str, rules: Iterable[tuple[Iterable[str], str]]) -> str | None:
    for keywords, variant_id in rules:
        if any(_keyword_in_text(text, kw) for kw in keywords):
            return variant_id
    return None


def _is_sending_donation_table(table_text: str) -> bool:
    return any(kw in table_text for kw in ("sending", "senden", "envoi", "envío", "invio", "wysyłka"))


# Product-specific variant keyword rules (order matters: first match wins)
_APM_VARIANTS: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "pay link",
            "pay links",
            "payment link",
            "payment links",
            "payment links and buttons",
            "buy buttons",
            "shopping cart buttons",
            "zahlungslink",
            "liens de paiement",
        ),
        "payment_links",
    ),
    (("cash a check", "cheque", "check"), "cash_a_check"),
    (("wire transfer", "virement", "transferencia bancaria"), "wire_transfer"),
    (("spendback", "remboursement"), "spendback_transfer"),
    (("debit card", "carte de débit", "tarjeta de débito"), "debit_card_transfer"),
    (("bank transfer", "domestic bank transfer", "virement bancaire"), "bank_transfer"),
    (("third-party digital wallet", "third party digital wallet", "third-party wallet"), "third_party_wallet"),
    (("foreign exchange", "fx spread", "currency conversion"), "fx_service"),
]

_ADVANCED_CARD_VARIANTS: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "eterminal",
            "terminal",
            "point of sale",
            "card present",
            "pagamenti telefonici",
            "telefonici",
            "pago por teléfono",
            "pagos por teléfono",
        ),
        "eterminal",
    ),
    (("standard credit", "carte standard", "tarjeta de crédito y débito"), "standard_card"),
    (("american express", "amex", "carte american express"), "american_express"),
    (
        (
            "advanced credit",
            "advanced debit",
            "carte bancaire avancés",
            "carte bancaire avancée",
            "cartes bancaires avancées",
            "pagamenti avanzati con carta",
            "avancerade betalningar med betalkort",
            "avancerat kredit- och betalkort",
        ),
        "advanced_card",
    ),
    (("payments advanced", "advanced payments", "payment advanced"), "payments_advanced"),
    (
        (
            "payments pro",
            "payment pro",
            "solution hébergée",
            "solution hébergée paypal",
            "hosted solution",
            "paypal pro",
            "pagamenti con paypal pro",
        ),
        "payments_pro",
    ),
    (("ach", "automated clearing", "addebito diretto", "sepa", "direktdebitering"), "ach"),
    (("additional risk", "risk factors", "risk factor", "chargeback protection", "fraud protection"), "risk_factors"),
    (("failure to implement", "express checkout", "checkout requis"), "express_checkout"),
    (("foreign exchange", "currency conversion", "devise", "fx as a service"), "fx_service"),
    (
        (
            "regroup",
            "flat rate",
            "forfait",
            "regroupée",
            "blended",
            "blandad prissättning",
            "piano tariffario misto",
            "combinada",
            "combinado",
            "tarifa combinada",
            "gecombineerde",
            "gecombineerd tarief",
            "kombinovanými sazbami",
            "kombinovanými sadzbami",
        ),
        "flat_rate",
    ),
    (("interchange plus plus", "interchange++"), "interchange_plus_plus"),
    (("interchange plus", "interchange+", "piano tariffario interchange plus"), "interchange_plus"),
]

_QR_BELOW_THRESHOLD: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "unter",
            "under",
            "below",
            "less than",
            "<",
            "bis zu",
            "up to",
            "jusqu'à",
            "inférieure",
            "inférieures",
            "inferior",
            "inferiores",
            "inferiori",
            "pari o inferiori",
            "a méně",
            "og derunder",
            "derunder",
            "og under",
            "under",
            "a menej",
            "i mniej",
            "lub mniej",
            "co najwyżej",
            "o menos",
            "o meno",
            "o mniej",
            "no máximo",
            "até",
            "al massimo",
            "fino a",
            "tai vähemmän",
            "vähemmän",
            "en minder",
            "minder",
            "eller mindre",
            "höchstens",
            "legfeljebb",
            "kevesebb",
            "hasta",
            "nejvýše",
            "nebo méně",
            "najviac",
            "alebo menej",
            "nižšej",
            "a nižšej",
            "ή λιγότερα",
            "λιγότερα",
            "και κάτω",
            "och lägre",
            "及以下",
            "或以下",
        ),
        "below_threshold",
    ),
)

_QR_ABOVE_THRESHOLD: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "over",
            "above",
            "greater than",
            ">",
            "mindestens",
            "at least",
            "à partir de",
            "au moins",
            "supérieure",
            "supérieures",
            "superior",
            "superiores",
            "superiori",
            "pari o superiori",
            "a více",
            "og derover",
            "derover",
            "og over",
            "over",
            "a viac",
            "i więcej",
            "lub więcej",
            "co najmniej",
            "o más",
            "o mas",
            "o più",
            "ou mais",
            "no mínimo",
            "pelo menos",
            "almeno",
            "più di",
            "tai enemmän",
            "enemmän",
            "en meer",
            "meer",
            "eller mer",
            "eller fler",
            "legalább",
            "több mint",
            "meghaladó",
            "a partir de",
            "desde",
            "nejméně",
            "nebo více",
            "najmenej",
            "alebo viac",
            "vyššej",
            "a vyššej",
            "ή περισσότερα",
            "περισσότερα",
            "και πάνω",
            "och högre",
            "及以上",
            "或以上",
        ),
        "above_threshold",
    ),
)

_MICROPAYMENT_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("mass payment", "mass payments"), "mass_payments"),
    (("digital", "digitala", "digitale", "dijital"), "digital_goods"),
)

_PAYPAL_CHECKOUT_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("crypto", "bitcoin", "cryptocurrency", "krypto", "cryptomonnaie", "criptomoneda"), "crypto"),
)

_OTHER_COMMERCIAL_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("campaign", "store cash", "campagne"), "campaign_fee"),
    (("pyusd",), "pyusd"),
    (("ach", "pay by bank", "virement bancaire"), "ach"),
    (
        (
            "card funded",
            "approvisionné par carte",
            "financiada",
            "financiado por cartão",
            "card-funded",
            "kortfinansierad",
            "kortilla rahoitettu",
            "kártyás kifizetések",
            "pagamento con carta",
            "kaartbetaling",
            "betaald met kaart",
            "kartou hrazená",
            "platba kartou",
            "płatność kartą",
            "płatności kartą",
        ),
        "card_funded",
    ),
)

_POS_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("qr code", "qr-code", "qr-kode", "qr kod"), "qr_code"),
    (
        (
            "manual",
            "manuelle",
            "manuale",
            "manual entry",
            "saisie manuelle",
            "saisie",
            "manuell inmatning",
            "manuel indtastning",
            "manuelle eingabe",
            "handmatig",
            "handmatig ingevoerd",
        ),
        "manual_entry",
    ),
    (
        (
            "card present",
            "present",
            "präsent",
            "présente",
            "presente",
            "transactions par carte",
            "transaktionen mit präsenter karte",
            "aktuella korttransaktioner",
            "korttransaktioner",
            "kortforevisning",
            "carta presente",
            "kaart aanwezig",
            "tilstedeværende kort",
        ),
        "card_present",
    ),
    (
        (
            "payment link",
            "zahlungslink",
            "zahlungslinks",
            "liens de paiement",
            "payment links",
            "betalningslänkar",
            "betalingslinks",
            "link di pagamento",
            "links de pagamento",
            "betaallinks",
            "betalingslink",
            "betalingslenker",
        ),
        "payment_links",
    ),
)

_DONATIONS_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "aufgeführte",
            "aufgelistet",
            "listed",
            "listed campaigns",
            "in elenco",
            "presenti in elenco",
            "indicadas",
            "indicada",
            "indicados",
            "indicado",
            "indiquées",
            "listées",
            "listado",
            "listados",
            "listada",
            "listadas",
        ),
        "campaign_listed",
    ),
    (
        (
            "nicht aufgeführte",
            "unlisted",
            "non listée",
            "non listados",
            "non listate",
        ),
        "campaign_unlisted",
    ),
    (
        (
            "campaign",
            "aktion",
            "collect",
            "campagne",
            "collecte",
            "cause",
            "dons collectifs",
            "fundraiser",
            "fundraisers",
            "fundraising",
            "recaudación",
            "recaudacion",
            "recaudación de fondos",
            "raccolta",
            "raccolta fondi",
            "collecte de fonds",
            "fundraising-kampagne",
        ),
        "campaign",
    ),
    (("button", "bouton", "botón", "pulsante", "knop"), "button"),
)

# APM variant rows whose labels describe a specific product/service rather than
# a list of methods should not carry extracted APM method tokens as unknown.
_APM_VARIANT_ONLY_VARIANTS: frozenset[str] = frozenset(
    {
        "payment_links",
        "cash_a_check",
        "wire_transfer",
        "spendback_transfer",
        "debit_card_transfer",
        "bank_transfer",
        "third_party_wallet",
        "fx_service",
    }
)

_INVOICE_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("rückzahlung", "repayment", "remboursement", "reembolso", "rimborso", "refund", "refund"), "repayment"),
)

_NONPROFIT_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("qr-code", "qr code", "qr-code-transaktionen", "qr-code-transaktion", "qr-code-zahlungen"), "qr_code"),
)

_PAY_LATER_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("payment link", "payment links", "zahlungslink", "liens de paiement"), "payment_links"),
)

_QR_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = _QR_BELOW_THRESHOLD + _QR_ABOVE_THRESHOLD

_DISPUTE_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("high volume", "high-volume", "hochvolumen", "grand volume", "alto volumen", "høj volumen"), "high_volume"),
    (("standard", "standart", "standard dispute"), "standard"),
)

_WITHDRAWAL_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ach", "automated clearing", "automated clearing house"), "ach"),
    (("wire transfer", "virement", "transferencia bancaria", "bank transfer"), "wire_transfer"),
    (("bank return", "return on withdrawal", "return on transfer", "returned"), "bank_return"),
    (("instant transfer", "instant bank transfer"), "instant_transfer"),
    (("bank account", "to a bank account"), "bank_account"),
    (("business debit", "business debit card", "bargeld", "geldautomat", "atm"), "business_debit_atm"),
    (("cards", "card"), "cards"),
    (("paypal payouts", "payouts"), "payouts"),
)

_SEPA_DIRECT_DEBIT_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Instant must be checked before standard because "Standardabrechnung" can
    # appear inside a cell that also mentions the instant option.
    (
        (
            "instant",
            "sofort",
            "sofortige",
            "sofortabrechnung",
            "instant settlement",
            "istantanea",
            "istantaneo",
        ),
        "instant_settlement",
    ),
    (("standard", "standardabrechnung", "standard settlement"), "standard_settlement"),
)

_FRAUD_PROTECTION_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("advanced", "professionelles tool", "professionelles"), "advanced"),
)

_RECORDS_REQUEST_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = ((("standard",), "standard"),)

_CARD_VERIFICATION_VARIANTS: tuple[tuple[tuple[str, ...], str], ...] = ((("standard",), "standard"),)

# Variant keyword rules for fixed/international surcharge schedule identity.
_VARIANT_RULES_BY_PRODUCT: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "advanced_card_payments": tuple(_ADVANCED_CARD_VARIANTS),
    "alternative_payment_methods": tuple(_APM_VARIANTS),
    "qr_code_payments": _QR_VARIANTS,
    "disputes": _DISPUTE_VARIANTS,
    "withdrawals": _WITHDRAWAL_VARIANTS,
    "sepa_direct_debit": _SEPA_DIRECT_DEBIT_VARIANTS,
    "fraud_protection": _FRAUD_PROTECTION_VARIANTS,
    "records_request": _RECORDS_REQUEST_VARIANTS,
    "card_verification": _CARD_VERIFICATION_VARIANTS,
}

# Variants that are considered "base" variants for a product. A fixed-fee
# table whose applicable variants include a base variant becomes the base
# schedule for that product; otherwise it is treated as a variant-specific
# schedule.
_BASE_VARIANTS_BY_PRODUCT: dict[str, frozenset[str]] = {
    "advanced_card_payments": frozenset({v for _, v in _ADVANCED_CARD_VARIANTS} | {"donations"}),
    "alternative_payment_methods": frozenset(
        {
            "default",
            "special",
            "bank_transfer",
            "debit_card_transfer",
            "spendback_transfer",
            "cash_a_check",
            "wire_transfer",
        }
    ),
    "other_commercial": frozenset({"standard", "campaign_fee", "pyusd", "ach", "card_funded"}),
    "paypal_checkout": frozenset({"standard", "venmo"}),
    "invoice_pay_later": frozenset({"standard", "payment_links"}),
    "qr_code_payments": frozenset({"standard"}),
    "micropayments": frozenset({"standard", "digital_goods", "mass_payments"}),
    "disputes": frozenset({"standard"}),
    "withdrawals": frozenset({"withdrawal", "bank_account", "cards"}),
    "sepa_direct_debit": frozenset({"standard_settlement"}),
    "fraud_protection": frozenset({"advanced"}),
    "records_request": frozenset({"standard"}),
    "card_verification": frozenset({"standard"}),
}


def _all_variant_matches(text: str, rules: Iterable[tuple[Iterable[str], str]]) -> list[str]:
    """Return all variant ids whose keywords appear in the normalized text."""
    seen: set[str] = set()
    result: list[str] = []
    for keywords, variant_id in rules:
        if any(_keyword_in_text(text, kw) for kw in keywords) and variant_id not in seen:
            seen.add(variant_id)
            result.append(variant_id)
    return result


def _applicable_variants_for_table(table: Table, base_name: str) -> list[str]:
    """Return the variant ids explicitly named in a schedule table caption."""
    rules = _VARIANT_RULES_BY_PRODUCT.get(base_name)
    if not rules:
        return []
    text = _table_text(table)
    return _all_variant_matches(text, rules)


def _variant_for_apm(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if any(m in _APM_SPECIAL_METHOD_IDS for m in methods):
        return "special"
    variant = _first_variant_match(norm_label, _APM_VARIANTS)
    if variant:
        return variant
    if _is_generic_apm_label(label):
        return "default"
    return "default"


def _variant_for_advanced_card(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_charity_label(combined):
        return "donations"
    if "american express" in norm_label or "americanexpress" in norm_label or "amex" in norm_label:
        return "american_express"
    return _first_variant_match(norm_label, _ADVANCED_CARD_VARIANTS) or "standard"


def _variant_for_qr_code(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _first_variant_match(norm_label, _QR_BELOW_THRESHOLD):
        return "below_threshold"
    if _first_variant_match(norm_label, _QR_ABOVE_THRESHOLD):
        return "above_threshold"
    return "standard"


def _variant_for_micropayments(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    variant = _first_variant_match(norm_label, _MICROPAYMENT_VARIANTS)
    if variant:
        return variant
    if is_intl:
        return "international"
    if is_dom:
        return "domestic"
    return "standard"


def _variant_for_paypal_checkout(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_charity_label(label):
        return "donations"
    parts = re.split(r"[^a-z0-9]+", norm_label)
    if "venmo" in parts and not any(
        i > 0 and parts[i - 1] in {"or", "and", "ou"} for i in range(1, len(parts)) if parts[i] == "venmo"
    ):
        return "venmo"
    variant = _first_variant_match(norm_label, _PAYPAL_CHECKOUT_VARIANTS)
    if variant:
        return variant
    if is_intl:
        return "international"
    if is_dom:
        return "domestic"
    return "standard"


def _variant_for_other_commercial(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_generic_other_commercial_label(label):
        return "standard"
    return _first_variant_match(norm_label, _OTHER_COMMERCIAL_VARIANTS) or "standard"


def _variant_for_pos_transactions(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(norm_label, _POS_VARIANTS) or "standard"


def _variant_for_donations(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_sending_donation_table(table_text):
        return "sending"
    unlisted_keywords = ("nicht aufgeführte", "unlisted", "non listée", "non listados", "non listate")
    if any(_keyword_in_text(norm_label, kw) for kw in unlisted_keywords):
        return "campaign_unlisted"
    return _first_variant_match(norm_label, _DONATIONS_VARIANTS) or "standard"


def _variant_for_nonprofit(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    variant = _first_variant_match(norm_label, _NONPROFIT_VARIANTS)
    if variant:
        return variant
    if "interchange" in norm_label:
        if "++" in norm_label or "plus plus" in norm_label or "interchange plus plus" in table_text:
            return "interchange_plus_plus"
        return "interchange_plus"
    return _first_variant_match(norm_label, _ADVANCED_CARD_VARIANTS) or "standard"


def _variant_for_invoice_pay_later(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(norm_label, _INVOICE_VARIANTS) or "standard"


def _variant_for_pay_later_consumer(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(norm_label, _PAY_LATER_VARIANTS) or "standard"


def _variant_for_withdrawals(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return (
        _first_variant_match(norm_label, _WITHDRAWAL_VARIANTS)
        or _first_variant_match(combined, _WITHDRAWAL_VARIANTS)
        or "standard"
    )


def _variant_for_sepa_direct_debit(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(combined, _SEPA_DIRECT_DEBIT_VARIANTS)


def _variant_for_fraud_protection(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(combined, _FRAUD_PROTECTION_VARIANTS) or "advanced"


def _variant_for_records_request(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(combined, _RECORDS_REQUEST_VARIANTS) or "standard"


def _variant_for_card_verification(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _first_variant_match(combined, _CARD_VERIFICATION_VARIANTS) or "standard"


_VARIANT_DISPATCH: dict[str, Callable[..., str | None]] = {
    "alternative_payment_methods": _variant_for_apm,
    "advanced_card_payments": _variant_for_advanced_card,
    "qr_code_payments": _variant_for_qr_code,
    "micropayments": _variant_for_micropayments,
    "paypal_checkout": _variant_for_paypal_checkout,
    "other_commercial": _variant_for_other_commercial,
    "pos_transactions": _variant_for_pos_transactions,
    "donations": _variant_for_donations,
    "nonprofit": _variant_for_nonprofit,
    "invoice_pay_later": _variant_for_invoice_pay_later,
    "pay_later_consumer": _variant_for_pay_later_consumer,
    "withdrawals": _variant_for_withdrawals,
    "sepa_direct_debit": _variant_for_sepa_direct_debit,
    "fraud_protection": _variant_for_fraud_protection,
    "records_request": _variant_for_records_request,
    "card_verification": _variant_for_card_verification,
}


def _variant_id_for_row(
    product_id: str,
    label: str,
    methods: list[str],
    table: Table | None = None,
    fee_text: str | None = None,
) -> str | None:
    """Return a stable variant id for a row, if needed."""
    norm_label = _norm(label)
    table_text = _table_text(table) if table else ""
    combined = norm_label + " " + table_text
    # Direct fixed-fee variants are often encoded in the fee cell text (e.g. two
    # SEPA settlement options in one cell), so include that context when it is
    # available.
    if product_id in _DIRECT_FIXED_FEE_PRODUCTS and fee_text:
        combined = combined + " " + _norm(fee_text)

    # Generic domestic/international variants are detected up-front so that
    # product-specific logic can be layered on top of them.
    is_international = _is_international_label(label)
    is_domestic = _is_domestic_label(label)

    resolver = _VARIANT_DISPATCH.get(product_id)
    if resolver:
        variant = resolver(label, norm_label, table_text, combined, methods, is_international, is_domestic)
        if variant is not None:
            return variant

    if is_international:
        return "international"
    if is_domestic:
        return "domestic"
    return None


def _is_charity_label(label: str) -> bool:
    """Return True if the label/table text indicates a charity/donation context."""
    text = _norm(label)
    return any(
        kw in text
        for kw in (
            "spende",
            "spenden",
            "donation",
            "donations",
            "donativ",
            "don de",
            "donazioni",
            "donativos",
            "caridad",
            "caridade",
            "charity",
            "charitable",
            "liefdadigheid",
            "liefdadigheids",
            "goede doel",
            "goede doelen",
            "välgörenhet",
            "jótékonysági",
            "φιλανθρωπ",
            "dons",
            "dona",
            "donação",
            "donações",
            "humanit",
            "non-profit",
            "nonprofit",
            "nonguvernamental",
        )
    )


def _is_generic_other_commercial_label(label: str) -> bool:
    """Return True if the label is a generic 'all other commercial' fallback."""
    text = _norm(label)
    return any(
        kw in text
        for kw in (
            "all other commercial",
            "alle anderen",
            "toutes les autres",
            "autres transactions",
            "autre transaction",
            "altre transazioni",
            "otras transacciones",
            "outras transações",
            "outros",
            "andre",
            "andere",
            "pozostałe",
            "transações comerciais",
            "business payments",
            "commercial transactions",
            "other commercial",
            "transacciones comerciales",
            "transactions commerciales",
        )
    )


def _is_generic_apm_label(label: str) -> bool:
    """Return True if the label is a generic 'all other APM' fallback."""
    text = _norm(label)
    return any(
        kw in text
        for kw in (
            "all other alternative payment",
            "alle anderen alternativen",
            "tous les autres moyens",
            "tous les autres modes",
            "tutti gli altri metodi",
            "todos los otros métodos",
            "todos os outros métodos",
            "alternative payment method",
            "alternative payment methods",
            "alternative zahlungsmethode",
            "alternative zahlungsmethoden",
            "autre moyen de paiement",
            "autres moyens de paiement",
            "autre mode de paiement",
            "metodo di pagamento alternativo",
            "metodi di pagamento alternativi",
            "método de pago alternativo",
            "métodos de pago alternativos",
            "andere betaalmethode",
            "andere betaalmethoden",
            "alternative betalingsmetode",
            "alternative betalingsmetoder",
            "alternative betaalmethode",
            "alternative betaalmethoden",
            "all other apm",
            "alle anderen apm",
            "apm",
            "abm",
        )
    )


def _extract_country_group_condition(label: str) -> dict[str, Any] | None:
    """Parse a row label like 'AG, BB, BM & SA' into a list of market codes.

    Returns an applies_to_markets condition if the label contains market codes.
    Generic default phrases are returned as the special 'all_other_markets' code.
    """
    text = label
    default_phrases = (
        "all other markets",
        "all other",
        "todos os outros",
        "tous les autres",
        "tutti gli altri",
        "alle anderen",
        "alla andra",
        "alle andre",
        "todos los demás",
        "overige",
        "pozostałe",
        "pozostale",
        "pozostalých",
        "egyeb",
        "altri",
        "sonstige",
    )
    if any(p in _norm(text) for p in default_phrases):
        return {"applies_to_markets": ["all_other_markets"]}
    # Look for 2-character uppercase market codes separated by commas,
    # ampersands, 'and' or whitespace. A 2-char code may be followed by
    # punctuation, not by another letter.
    matches = re.findall(r"(?<![A-Za-z0-9])([A-Z]{2})(?![A-Za-z0-9])", text)
    # Filter out a few common false positives.
    codes = [m for m in matches if m not in {"QR", "ON"}]
    if not codes:
        return None
    return {"applies_to_markets": sorted(set(codes))}


def _pricing_plan_for_label(label: str) -> str | None:
    """Detect blended/standard/interchange pricing plan from a row label."""
    text = _norm(label)
    if "interchange plus plus" in text or "interchange++" in text:
        return "interchange_plus_plus"
    if "interchange plus" in text:
        return "interchange_plus"
    if any(
        kw in text
        for kw in (
            "blended",
            "regroupée",
            "regroup",
            "flat rate",
            "forfait",
            "misto",
            "tariffario misto",
            "piano tariffario misto",
            "blandad prissättning",
            "combinada",
            "combinado",
            "tarifa combinada",
            "gecombineerde",
            "gecombineerd tarief",
            "kombinovanými sazbami",
            "kombinovanými sadzbami",
        )
    ):
        return "blended"
    if "standard paypal payment" in text:
        return "standard"
    return None


def _card_payment_methods_from_label(label: str) -> list[str] | None:
    """Extract card brand names listed in an advanced card row label."""
    text = _norm(label)
    methods: list[str] = []
    for keyword, method_id in (
        ("visa", "visa"),
        ("mastercard", "mastercard"),
        ("maestro", "maestro"),
        ("china unionpay", "china_union_pay"),
        ("cup", "china_union_pay"),
        ("diners", "diners"),
        ("discover", "discover"),
        ("jcb", "jcb"),
        ("cofidis", "cofidis"),
        ("cetelem", "cetelem"),
        ("cofinoga", "cofinoga"),
        ("carte bancaire", "carte_bancaire"),
        ("debit card", "debit_card"),
        ("credit card", "credit_card"),
        ("other card", "other_card"),
    ):
        if _keyword_in_text(text, keyword) and method_id not in methods:
            methods.append(method_id)
    return methods if methods else None


def _service_for_donation_label(label: str) -> str | None:
    """Map a donation row label to its underlying service, if any."""
    text = _norm(label)
    if any(
        kw in text
        for kw in (
            "website payments pro",
            "payments pro",
            "solution hébergée",
            "paypal pro",
            "pagamenti con paypal pro",
            "hosted solution",
        )
    ):
        return "website_payments_pro"
    if any(kw in text for kw in ("virtual terminal", "eterminal", "e-terminal", "pagamenti telefonici", "telefonici")):
        return "virtual_terminal"
    if any(
        kw in text
        for kw in (
            "advanced credit",
            "advanced debit",
            "avancerat kredit",
            "carte bancaire avancés",
            "pagamenti avanzati con carta",
            "avancerade betalningar med betalkort",
            "avancerat kredit- och betalkort",
        )
    ):
        return "advanced_card"
    return None


def _conditions_for_apm(
    conditions: dict[str, Any],
    label: str,
    methods: list[str] | None,
    variant_id: str | None,
) -> None:
    """Populate conditions for an alternative payment methods row."""
    if methods is None:
        methods, _ = _extract_apm_methods(label)
    if methods:
        conditions["payment_methods"] = sorted(methods)
    if variant_id == "third_party_wallet":
        conditions["payment_methods"] = ["third_party_wallet"]
    if variant_id == "fx_service":
        conditions["service"] = "foreign_exchange"


def _conditions_for_donations(
    conditions: dict[str, Any],
    product_id: str,
    label: str,
) -> None:
    """Populate conditions for a donations-related row."""
    conditions["transaction_purpose"] = "donation"
    if product_id in ("advanced_card_payments", "nonprofit"):
        service = _service_for_donation_label(label)
        if service:
            conditions["service"] = service


def _service_for_advanced_card(label: str, variant_id: str) -> str | None:
    """Return the service condition for advanced-card rows, if any."""
    if variant_id == "fx_service":
        text = _norm(label)
        if "spread" in text:
            return "fx_spread"
        if "as a service" in text:
            return "fx_as_a_service"
    return None


def _conditions_for_advanced_card(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
) -> None:
    """Populate conditions for an advanced card or nonprofit card row."""
    if variant_id == "eterminal":
        conditions["authorization_channel"] = "terminal"
        conditions["point_of_sale"] = True
    if variant_id.startswith("interchange_plus"):
        conditions["pricing_plan"] = variant_id
    else:
        plan = _pricing_plan_for_label(label)
        if plan:
            conditions["pricing_plan"] = plan
    service = _service_for_advanced_card(label, variant_id)
    if service:
        conditions["service"] = service
    if variant_id == "american_express":
        conditions["payment_methods"] = ["american_express"]
    else:
        card_methods = _card_payment_methods_from_label(label)
        if card_methods:
            conditions["payment_methods"] = card_methods


def _conditions_for_pos(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
) -> None:
    """Populate conditions for a POS transaction row."""
    if variant_id == "card_present":
        conditions["card_present"] = True
        conditions["point_of_sale"] = True
        conditions["authorization_channel"] = "terminal"
    elif variant_id == "manual_entry":
        conditions["card_present"] = False
        conditions["authorization_channel"] = "manual"
    elif variant_id == "qr_code":
        conditions["payment_methods"] = ["qr_code"]
        conditions["point_of_sale"] = True
    elif variant_id == "payment_links":
        text = _norm(label)
        if any(kw in text for kw in ("paypal checkout", "venmo", "pay later", "guest checkout")):
            conditions["payment_methods"] = sorted(["paypal_checkout", "venmo", "pay_later", "guest_checkout"])
        elif any(
            kw in text
            for kw in ("standard credit", "debit card", "apple pay", "third-party wallets", "third party wallets")
        ):
            conditions["payment_methods"] = sorted(["card", "apple_pay", "third_party_wallet"])


def _conditions_for_paypal_checkout(
    conditions: dict[str, Any],
    variant_id: str,
) -> None:
    """Populate conditions for a PayPal Checkout row."""
    if variant_id == "venmo":
        conditions["payment_methods"] = ["venmo"]
    elif variant_id == "crypto":
        conditions["payment_methods"] = ["cryptocurrency"]


def _service_for_other_commercial_ach(table: Table | None) -> str | None:
    """Return the service indicated by an other-commercial ACH table heading."""
    if not table:
        return None
    table_text = _norm(_table_text(table))
    if "invoic" in table_text:
        return "invoicing"
    if "online" in table_text and ("card" in table_text or "payment" in table_text):
        return "online_payments"
    return None


def _conditions_for_other_commercial(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
    table: Table | None,
) -> None:
    """Populate conditions for an other-commercial row."""
    if variant_id == "pyusd":
        conditions["pricing_plan"] = "pyusd"
    elif variant_id == "ach":
        conditions["payment_methods"] = ["ach"]
        service = _service_for_other_commercial_ach(table)
        if service:
            conditions["service"] = service
    elif variant_id == "card_funded":
        conditions["funding_source"] = "card"


def _transaction_region_for_variant(
    label: str,
    variant_id: str | None,
    table: Table | None,
) -> str | None:
    """Infer a transaction_region from the variant, row label, or table caption."""
    if variant_id in ("domestic", "international"):
        return variant_id
    if variant_id in ("crypto", "digital_goods"):
        if _is_international_label(label):
            return "international"
        if _is_domestic_label(label):
            return "domestic"
        return None
    if _is_international_label(label) and not _is_domestic_label(label):
        return "international"
    if _is_domestic_label(label) and not _is_international_label(label):
        return "domestic"
    if table:
        table_text = _table_text(table)
        if _is_international_label(table_text) and not _is_domestic_label(table_text):
            return "international"
        if _is_domestic_label(table_text) and not _is_international_label(table_text):
            return "domestic"
    return None


def _conditions_for_row(
    product_id: str,
    variant_id: str | None,
    label: str,
    methods: list[str] | None = None,
    table: Table | None = None,
) -> dict[str, Any]:
    """Return calculable conditions for a product rule based on the source row."""
    conditions: dict[str, Any] = {}
    if product_id == "nonprofit":
        conditions["merchant_approval_required"] = True
    if product_id == "alternative_payment_methods":
        _conditions_for_apm(conditions, label, methods, variant_id)
    if product_id == "donations" or variant_id == "donations":
        _conditions_for_donations(conditions, product_id, label)
    if product_id in ("advanced_card_payments", "nonprofit") and variant_id:
        _conditions_for_advanced_card(conditions, label, variant_id)
    if product_id == "pos_transactions" and variant_id:
        _conditions_for_pos(conditions, label, variant_id)
    if product_id == "paypal_checkout" and variant_id:
        _conditions_for_paypal_checkout(conditions, variant_id)
    if product_id == "other_commercial" and variant_id:
        _conditions_for_other_commercial(conditions, label, variant_id, table)
    if product_id == "withdrawals" and variant_id and variant_id != "standard":
        conditions["withdrawal_method"] = variant_id
    region = _transaction_region_for_variant(label, variant_id, table)
    if region:
        conditions["transaction_region"] = region
    market_condition = _extract_country_group_condition(label)
    if market_condition:
        conditions.update(market_condition)
    if product_id == "qr_code_payments" and variant_id:
        amount_condition = _extract_amount_condition(label)
        if amount_condition:
            conditions["amount"] = amount_condition
    return conditions


def _maximum_fee_schedule_for_conditions(conditions: dict[str, Any]) -> str | None:
    """Map withdrawal/payout conditions to the corresponding max-fee schedule."""
    if conditions.get("transaction_region") == "international":
        return "payouts_international"
    if conditions.get("transaction_region") == "domestic":
        return "payouts_domestic"
    if conditions.get("applies_to_markets") == ["US"]:
        return "payouts_us"
    return None


def _extract_amount_condition(label: str) -> dict[str, Any] | None:
    """Parse a threshold expression like 'below 10.00 EUR' into a condition."""
    text = _norm(label)
    operators = {
        "<": "lt",
        "<=": "lte",
        ">": "gt",
        ">=": "gte",
        "under": "lt",
        "below": "lt",
        "less than": "lt",
        "unter": "lt",
        "bis zu": "lt",
        "up to": "lt",
        "jusqu'à": "lt",
        "inférieure": "lt",
        "inférieures": "lt",
        "inferior": "lt",
        "über": "gt",
        "over": "gt",
        "above": "gt",
        "greater than": "gt",
        "mindestens": "gt",
        "at least": "gt",
        "à partir de": "gt",
        "supérieure": "gt",
        "supérieures": "gt",
        "supérieure ou égale": "gte",
        "supérieures ou égales": "gte",
        "superior": "gt",
    }
    for op_token, op in operators.items():
        # Match the operator token followed by a number and optional currency.
        pattern = re.escape(op_token) + r"\s+([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z]{3})?"
        match = re.search(pattern, text)
        if match:
            value = normalize_decimal_string(match.group(1))
            currency = match.group(2)
            result: dict[str, Any] = {"operator": op, "value": value}
            if currency:
                result["currency"] = currency.upper()
            return result
    return None


# Schedule captions that should be treated as advanced_card_payments schedules
# before the generic online_card_payments mapping takes precedence.
_ADVANCED_CARD_SCHEDULE_KEYWORDS: tuple[str, ...] = (
    "advanced credit and debit card payments",
    "advanced card",
    "payments advanced",
    "payments pro",
    "virtual terminal",
    "eterminal",
    "e-terminal",
    "paypal intégral évolution",
    "intégral évolution",
    "interchange plus plus",
    "interchange plus",
    "paypal card payment services",
)


_SCHEDULE_NAME_FROM_TABLE_MAPPING = {
    "goods_and_services": (
        "geld für waren und dienstleistungen",
        "waren und dienstleistungen",
        "goods and services",
        "varer og tjenesteydelser",
        "varer og tjenester",
        "bienes y servicios",
        "beni e servizi",
        "goederen en diensten",
        "produkter och tjänster",
        "produkter og tjenester",
    ),
    "donations": (
        "spende",
        "donation",
        "donationer",
        "donations",
        "donaties",
        "donativos",
        "donativas",
        "doações",
        "donazioni",
        "lahjoitukset",
        "darowizn",
        "príspevkov",
        "príspevky",
        "adományok",
        "δωρεές",
        "δωρεες",
        "donatii",
        "дарения",
        "donacije",
        "donacijo",
        "příspěvky",
        "příspěvků",
    ),
    "nonprofit": (
        "gemeinnützig",
        "nonprofit",
        "non-profit",
        "velgørende",
        "charity",
        "charitable",
        "charitat",
        "caridad",
        "caridade",
        "caritative",
        "organizaciones sin fines de lucro",
        "organizaciones benéficas",
        "organización benéfica",
        "benéfica",
        "benéficas",
        "organizzazioni senza scopo di lucro",
        "organisasjoner",
        "organizacja non-profit",
        "organizacje charytatywne",
        "charitatívnych",
        "charitativních",
        "charitativní",
        "jótékonysági",
        "välgörenhetsorganisationer",
        "välgörenhet",
        "φιλανθρωπικές",
        "φιλανθρωπικά",
        "φιλανθρωπικού",
        "instituições de solidariedade",
        "instituição de caridade",
        "institución de solidaridad",
        "entidades sin ánimo de lucro",
        "associazioni di volontariato",
        "enti benefici",
        "ente benefico",
        "a favore di enti benefici",
        "liefdadigheid",
    ),
    "micropayments": (
        "mikrozahlung",
        "micropayment",
        "mikrobetaling",
        "mikrobetalinger",
        "microbetaling",
        "microbetalingen",
        "mikromaksu",
        "mikromaksut",
        "mikropłatność",
        "mikropłatności",
        "micropagos",
        "micropaiement",
        "micropaiements",
        "micropagamentos",
        "mikrobetalning",
        "mikrobetalningar",
        "mikroplatby",
        "mikroπληρωμές",
        "μικροπληρωμές",
        "μικροπληρωμες",
        "mikrotransakciók",
        "mikrotransakcije",
        "mikrokifizetés",
        "mikrokifizetések",
        "micropagamenti",
        "小額付款",
        "小额付款",
        "小額支付",
        "小额支付",
    ),
    "alternative_payment_methods": (
        "alternative zahlungsmethode",
        "alternative payment",
        "apm",
        "alternativ betalingsmetode",
        "alternative betalingsmetode",
        "alternative betaalmethode",
        "métodos de pago alternativos",
        "metodi di pagamento alternativi",
        "metodo di pagamento alternativo",
        "andere betaalmethode",
        "autre moyen de paiement",
        "autre mode de paiement",
        "alternatív fizetési",
        "alternatívny spôsob platby",
        "alternativní způsob platby",
        "alternatywna forma płatności",
        "alternativ betalningsmetod",
        "alternativ betalingsmåte",
        "vaihtoehtoinen maksutapa",
        "vmt",
        "abm",
        "εναλλακτικός τρόπος πληρωμής",
        "εναλλακτικος τροπος πληρωμης",
        "συναλλαγές με εμπ",
        "συναλλαγες με εμπ",
    ),
    "online_card_payments": (
        "online-kartenzahlungen",
        "online card",
        "online card payments",
        "online-kortbetaling",
        "online kort",
        "online kortbetalingstjenester",
        "kortbetalingstjenester",
        "kortbetalningstjänster",
        "kortbetalning",
        "tarjetas de crédito y débito",
        "kredit- och debitkort",
        "kredit- og betalingskort",
        "credit and debit card",
        "servizi di pagamento con carta",
        "services de paiement par carte",
        "serviços de pagamento com cartão",
        "servicios de pago con tarjeta",
        "online platby kartou",
        "online kártyás",
        "online betaalservices",
        "transacties ontvangen via online betaalservices",
        "pago por internet",
        "servicios de pago por internet",
        "płatności online kartą",
        "usług płatności online kartą",
        "verkkokorttimaksupalvelut",
        "verkkokorttimaksu",
        "ηλεκτρονικές πληρωμές με κάρτα",
        "ηλεκτρονικες πληρωμες με καρτα",
        "υπηρεσιών paypal για ηλεκτρονικές",
        "υπηρεσιων paypal για ηλεκτρονικες",
    ),
    "pos_transactions": (
        "point of sale",
        "präsenter karte",
        "kortforevisning",
        "card present",
        "ponto de venda",
        "punto de venta",
        "punto vendita",
        "punkty sprzedaży",
    ),
    "qr_code_payments": (
        "qr",
        "qr-code",
        "qr code",
        "qr code payments",
        "qr-code-transaktionen",
        "qr-code-transaktion",
        "qr-code-zahlungen",
        "qr-code-zahlung",
        "qr-code-betalinger",
        "qr kode",
        "qr-kode",
        "qr kode-betalinger",
        "qr-kode-betalinger",
        "kódů qr",
        "qr kódů",
        "kódy qr",
        "qr kódy",
        "código qr",
        "códigos qr",
    ),
    "invoice_pay_later": (
        "invoicing",
        "invoicing transaction",
        "invoice",
        "rechnung",
        "rechnungen",
        "facture",
        "facturas",
        "fattura",
        "fatture",
        "factuur",
        "faktura",
        "faktury",
        "faktur",
        "számla",
        "fakturor",
    ),
    "advanced_card_payments": (
        "advanced credit and debit card payments",
        "advanced credit",
        "advanced debit",
        "payments advanced",
        "payments pro",
        "pasarela integral",
        "virtual terminal",
        "eterminal",
        "e-terminal",
        "paypal intégral évolution",
        "intégral évolution",
        "pago por teléfono",
        "pagos por teléfono",
        "pagamento telefonico",
        "pagamenti telefonici",
        "servizi telefonici",
        "telefónico",
        "telefónica",
        "telefonisch",
        "telefonische",
        "téléphonique",
        "téléphoniques",
        "telefonico",
        "telefonica",
        "paypal pro",
        "interchange plus plus",
        "interchange plus",
        "payPal card payment services",
        "card payment services",
        "erweiterte kredit- und debitkartenzahlungen",
        "zahlungen mit kredit- und debitkarten mit erweiterten funktionen",
        "kredit- und debitkarten mit erweiterten funktionen",
        "advanced card",
        "erweiterte kartenzahlung",
        "kredit- og betalingskort",
        "kredit- og debitkort",
        "kredit- och debitkort",
        "kreditkort",
        "credit and debit card",
        "servizi di pagamento con carta",
        "services de paiement par carte",
        "serviços de pagamento com cartão",
        "servicios de pago con tarjeta",
        "online platby kartou",
        "online kártyás",
        "verkkokorttimaksupalvelut",
        "verkkokorttimaksu",
        "ηλεκτρονικές πληρωμές με κάρτα",
        "ηλεκτρονικες πληρωμες με καρτα",
    ),
    "recipient_service": (
        "recipient service",
        "recipients of eea",
        "recipients of",
        "recipient of",
        "eea based",
        "empfänger",
        "empfängerinnen",
        "ontvanger",
        "destinatario",
        "destinataire",
        "grand-bretagne",
        "großbritannien",
        "united kingdom",
        "in großbritannien ansässig",
        "british recipient",
        "uk recipient",
        "příjemce",
        "příjemců",
        "příjemci",
        "příjemcům",
        "príjemca",
        "príjemcov",
        "príjemcom",
        "odběratel",
        "odběratele",
        "velké británii",
        "velká británie",
        "velká británia",
        "wielka brytania",
        "wielkiej brytanii",
        "regatul unit",
        "marea britanie",
        "wielkiej brytanii",
        "fogadó",
        "fogadó felek",
        "címzett",
        "egyesült királyság",
        "egyesült királyságban",
        "mottagare",
        "mottakere",
        "mottaker",
        "storbritannien",
        "storbritannia",
        "regno unito",
        "destinatari",
        "παραλήπτες",
        "παραλήπτης",
        "ηνωμένο βασίλειο",
        "ηνωμένου βασιλείου",
        "spojeného kráľovstva",
        "spojené kráľovstvo",
        "für empfänger",
        "für empfängerinnen",
        "aus großbritannien",
        "from the united kingdom",
        "united kingdom based",
        "united kingdom-based",
    ),
    "withdrawals": (
        "withdrawal",
        "withdrawals",
        "withdraw",
        "auszahlung",
        "auszahlungen",
        "payout",
        "payouts",
        "uttag",
        "uitoog",
        "wypłata",
        "retrait",
        "retiro",
        "ritiro",
        "bank transfer",
        "bank transfer withdrawal",
        "transfer to card",
        "transfer to a card",
        "μεταφορά σε κάρτα",
        "transfert sur carte",
        "transferencia a tarjeta",
        "trasferimento su carta",
        "überweisung auf karte",
        "przelew na kartę",
        "převod na kartu",
        "prevod na kartu",
        "transfer do karty",
        "kártyára történő átutalás",
        "transfer till kort",
        "overførsel til kort",
        "disbursement",
        "disbursements",
        "wire transfer",
        "wire transfer disbursement",
        "abbuchen",
        "guthaben von einem paypal-geschäftskonto abbuchen",
    ),
    "chargebacks": (
        "chargeback",
        "chargebacks",
        "rückbuchung",
        "rückbuchungen",
        "rückbuchungsgebühr",
        "contra reembolso",
        "chargeback fee",
        "chargeback fees",
    ),
    "refunds": (
        "refund",
        "refunds",
        "rückerstattung",
        "rückerstattungen",
        "erstattung",
        "erstattungen",
        "terugbetaling",
        "remboursement",
        "rimborso",
        "reembolso",
    ),
    "disputes": (
        "dispute",
        "disputes",
        "streit",
        "claim",
        "claims",
        "disputed",
        "klage",
        "beschwerde",
        "geschil",
        "litige",
        "controversia",
        "contestazione",
    ),
    "card_verification": (
        "card verification",
        "kartenverifizierung",
        "kartenbestätigung",
        "kreditkartenbestätigung",
        "credit card verification",
        "debit card verification",
        "3d secure",
        "verification",
        "verifizierung",
    ),
    "commercial": (
        "geschäftlichen transaktionen",
        "commercial transaction",
        "commercial",
        "erhvervsbetalinger",
        "erhverv",
        "business payments",
        "standardgebyr",
        "standardavgift",
        "standaardtarief",
        "tarifa estándar",
        "tarifa standard",
        "tarifa padrão",
        "tariffa standard",
        "tarification standard",
        "standard fee",
        "standard rate",
        "standardní sazba",
        "štandardná sadzba",
        "szokásos díja",
        "szokásos díj",
        "standardowa stawka",
        "standardtaxa",
        "standardsats",
        "standard taxa",
        "εμπορικές συναλλαγές",
        "εμπορικες συναλλαγες",
        "commerciale",
        "commerciële",
        "commerciale",
        "kommersielle",
        "kommercielle",
        "kommersiella",
        "komercyjne",
        "komerčních",
        "komerčných",
        "kereskedelmi",
        "comerciale",
        "obchodní transakce",
        "obchodné transakcie",
        "emporia",
    ),
}


def _schedule_name_from_table(table: Table, default: str | None) -> str:
    text = _table_text(table)
    if any(kw in text for kw in _ADVANCED_CARD_SCHEDULE_KEYWORDS):
        return "advanced_card_payments"
    mapping = _SCHEDULE_NAME_FROM_TABLE_MAPPING
    for name, keywords in mapping.items():
        for kw in keywords:
            if _norm(kw) in text:
                return name
    return default or "commercial"


def _signature_key(signature: dict[str, Any]) -> frozenset[tuple[str, Any]]:
    """Return a hashable representation of a schedule signature for grouping."""
    items: list[tuple[str, Any]] = []
    for k, v in signature.items():
        if isinstance(v, list):
            v = tuple(sorted(v))
        elif isinstance(v, dict):
            v = tuple(sorted((kk, tuple(sorted(vv)) if isinstance(vv, list) else vv) for kk, vv in v.items()))
        items.append((k, v))
    return frozenset(items)


def _schedule_signature_for_row(
    row: Row,
    base_name: str,
    table_text: str = "",
    use_row_label: bool = True,
) -> dict[str, Any]:
    """Return applicability dimensions encoded in a schedule table row and context.

    Schedule tables usually contain currency names as row labels, so market/amount
    applicability is read from the table heading/caption by default.  Rate-table
    rows (``use_row_label=True``) may encode applicability in the label itself.
    """
    label = _row_label(row) if use_row_label else ""
    combined_text = f"{label} {table_text}".strip()
    sig: dict[str, Any] = {}
    market_condition = _extract_country_group_condition(label) or _extract_country_group_condition(table_text)
    if market_condition:
        sig["applies_to_markets"] = market_condition["applies_to_markets"]
    # QR threshold variants are derived from the table caption via
    # _applicable_variants_for_table, so the amount_tier is encoded in the
    # base schedule id and should not be duplicated in the signature.
    if base_name == "advanced_card_payments" and "interchange" in _norm(combined_text):
        if "plus plus" in _norm(combined_text) or "++" in combined_text:
            sig["pricing_plan"] = "interchange_plus_plus"
        else:
            sig["pricing_plan"] = "interchange_plus"
    return sig


def _schedule_suffix_from_signature(signature: dict[str, Any]) -> str:
    """Canonical string suffix for a schedule applicability signature."""
    parts: list[str] = []
    for key in sorted(signature):
        value = signature[key]
        if key == "applies_to_markets":
            if isinstance(value, list):
                if value == ["all_other_markets"] or not value:
                    continue
                parts.append("applies_to_markets=" + "_".join(sorted(value)).lower())
            else:
                parts.append(f"applies_to_markets={_norm(str(value))}")
        else:
            parts.append(f"{key}={_norm(str(value))}")
    return "__".join(parts)


def _schedule_id(base_id: str, signature: dict[str, Any]) -> str:
    """Build a schedule id from a base id and an applicability signature."""
    suffix = _schedule_suffix_from_signature(signature)
    if not suffix:
        return base_id
    return f"{base_id}__{suffix}"


def _select_schedule_id(
    base_id: str,
    signature: dict[str, Any],
    available: dict[str, Any],
    fallback_bases: tuple[str, ...] = (),
    product_base: str | None = None,
) -> tuple[str | None, str | None]:
    """Return the schedule id a rule should reference and an optional inheritance source.

    Resolution priority is:

    1. Exact variant-specific schedule (with the full signature suffix).
    2. Direct product-family base schedule (with the same suffix or as a base).
    3. Explicitly proven cross-product inheritance from ``fallback_bases``.
    4. ``None`` if no usable schedule exists.

    The direct product-family base is only consulted when it differs from the
    variant-specific ``base_id`` so that variant-specific schedules are tried
    first, but a generic product schedule always wins over cross-product
    inheritance.
    """
    suffix = _schedule_suffix_from_signature(signature)
    candidates: list[str] = []
    if suffix:
        candidates.append(f"{base_id}__{suffix}")
    candidates.append(base_id)
    if product_base and product_base != base_id:
        if suffix:
            candidates.append(f"{product_base}__{suffix}")
        candidates.append(product_base)
    intended = candidates[0] if candidates else None
    for candidate in candidates:
        if candidate in available:
            return candidate, None
    # No existing schedule. Report the intended id and the first available
    # fallback source schedule so that explicit inheritance can be attempted.
    if fallback_bases:
        for fallback in fallback_bases:
            if suffix:
                suffixed = f"{fallback}__{suffix}"
                if suffixed in available:
                    return intended, suffixed
                if fallback in available:
                    return intended, fallback
            elif fallback in available:
                return intended, fallback
    return None, None


def _signature_from_conditions(conditions: dict[str, Any], base_id: str, product_id: str) -> dict[str, Any]:
    """Build a schedule applicability signature from rule conditions."""
    sig: dict[str, Any] = {}
    markets = conditions.get("applies_to_markets")
    if markets:
        sig["applies_to_markets"] = markets
    amount = conditions.get("amount")
    if isinstance(amount, dict):
        op = amount.get("operator")
        if op in {"lt", "lte", "under", "below", "less than", "up to"}:
            sig["amount_tier"] = "below_threshold"
        elif op in {"gt", "gte", "above", "over", "greater than", "at least", "mindestens"}:
            sig["amount_tier"] = "above_threshold"
    pricing_plan = conditions.get("pricing_plan")
    if pricing_plan and not base_id.endswith(str(pricing_plan)):
        sig["pricing_plan"] = pricing_plan
    return sig


def _conditions_from_schedule_id(schedule_id: str) -> dict[str, Any]:
    """Parse an applicability suffix from a schedule id into conditions."""
    conditions: dict[str, Any] = {}
    if "__" not in schedule_id:
        return conditions
    _, suffix = schedule_id.split("__", 1)
    for part in suffix.split("__"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "applies_to_markets":
            if value == "all_other_markets":
                conditions[key] = ["all_other_markets"]
            else:
                conditions[key] = sorted(value.split("_"))
        elif key == "amount_tier" or key == "pricing_plan" or key == "transaction_region":
            conditions[key] = value
    return conditions


# ---------------------------------------------------------------------------
# Schedule extraction
# ---------------------------------------------------------------------------


def _extract_fixed_fee_schedule(
    table: Table, base_name: str, source: Source | None = None
) -> dict[str, FixedFeeSchedule]:
    """Extract fixed-fee schedules grouped by row applicability signature."""
    table_text = _table_context_original(table)
    groups: dict[frozenset[tuple[str, Any]], dict[str, str]] = {}
    group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
    # Determine which row cells are charge/fee columns. The first column is
    # normally the label/currency name. Any cells beyond the header count are
    # stray cells (e.g. footnotes or converted amounts) and must be ignored.
    header_count = len(table.headers)
    if header_count:
        charge_indices = [i for i in range(1, header_count) if table.headers[i].text.strip()]
    else:
        charge_indices: list[int] = []
    for row in table.rows:
        if _is_limit_or_cap_row(_row_label(row), _row_fee_cell(row)):
            continue
        signature = _schedule_signature_for_row(row, base_name, table_text, use_row_label=False)
        key = _signature_key(signature)
        group_keys.setdefault(key, signature)
        amounts = groups.setdefault(key, {})
        cells = row.cells
        iterable = (
            (cells[i] for i in charge_indices if i < len(cells)) if charge_indices else cells[1:]
        )  # skip the label cell
        for cell in iterable:
            if not cell.text.strip():
                continue
            money = _cell_money(cell)
            if money:
                amounts[money[0]] = money[1]
                continue
            # Some cells contain templated placeholders like {{...}}; skip them.
            if "{{" in cell.text:
                continue
            # Fallback: parse an explicit "amount CUR" text.
            parts = cell.text.strip().split()
            if len(parts) >= 2 and parts[-1].upper() in CURRENCY_CODES:
                with contextlib.suppress(ValueError):
                    amounts[parts[-1].upper()] = normalize_decimal_string(parts[0])
    if not groups:
        return {}

    # Preserve per-fragment provenance.  Rows are already tagged with their
    # original document/component id by the components extractor.
    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    result: dict[str, FixedFeeSchedule] = {}
    for key, amounts in groups.items():
        if not amounts:
            continue
        suffix = _schedule_suffix_from_signature(group_keys[key])
        result[suffix] = FixedFeeSchedule(entries=amounts, sources=sources)
    return result


def _extract_maximum_fee_schedule(table: Table, source: Source | None = None) -> dict[str, FixedFeeSchedule]:
    """Extract per-region, per-applicability maximum-fee-cap schedules.

    Tables such as "Fee and maximum fee cap for PayPal Payouts" contain a
    currency column and several columns for different max-fee caps. Each cap
    column becomes a separate ``payouts_<region>`` schedule, further split by
    the row's market group or amount tier.
    """
    schedules: dict[str, FixedFeeSchedule] = {}
    if not table.headers:
        return schedules

    table_text = _table_context_original(table)
    for col_idx in range(1, len(table.headers)):
        base_id = _max_fee_base_id(table.headers[col_idx].text)
        if not base_id:
            continue

        groups: dict[frozenset[tuple[str, Any]], dict[str, str]] = {}
        group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
        for row in table.rows:
            cells = [c for c in row.cells if c.text.strip()]
            if col_idx >= len(cells):
                continue
            money = _max_fee_money(cells[0], cells[col_idx])
            if not money:
                continue
            currency_code, amount = money
            signature = _schedule_signature_for_row(row, base_id, table_text, use_row_label=False)
            key = _signature_key(signature)
            group_keys.setdefault(key, signature)
            amounts = groups.setdefault(key, {})
            amounts[currency_code] = amount

        if not groups:
            continue

        sources = _max_fee_schedule_sources(table, source)
        for key, entries in groups.items():
            if not entries:
                continue
            schedule_id = _schedule_id(base_id, group_keys[key])
            _merge_max_fee_schedule(schedules, schedule_id, entries, sources)

    return schedules


def _max_fee_base_id(header: str) -> str | None:
    """Map a maximum-fee schedule header to its base schedule id."""
    header_norm = _norm(header)
    if "maximum fee cap" not in header_norm and "max fee cap" not in header_norm:
        return None
    if "us" in header_norm:
        return "payouts_us"
    if "domestic" in header_norm:
        return "payouts_domestic"
    if "international" in header_norm:
        return "payouts_international"
    return None


def _max_fee_money(currency_cell: Any, amount_cell: Any) -> tuple[str, str] | None:
    """Extract (currency_code, amount) from a maximum-fee schedule row."""
    money = _cell_money(amount_cell)
    if not money:
        parts = amount_cell.text.strip().split()
        if len(parts) >= 2 and parts[-1].upper() in CURRENCY_CODES:
            with contextlib.suppress(ValueError):
                money = (parts[-1].upper(), normalize_decimal_string(parts[0]))
        else:
            return None
    if not money:
        return None
    currency = _cell_money(currency_cell)
    if currency:
        return currency[0], money[1]
    return money


def _max_fee_schedule_sources(table: Table, source: Source | None) -> list[Provenance]:
    """Build provenance sources for a maximum-fee schedule."""
    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    return sources


def _merge_max_fee_schedule(
    schedules: dict[str, FixedFeeSchedule],
    schedule_id: str,
    entries: dict[str, str],
    sources: list[Provenance],
) -> None:
    """Merge a group of maximum-fee entries into an existing or new schedule."""
    existing = schedules.get(schedule_id)
    if existing:
        merged_entries = dict(existing.entries)
        for currency, amount in entries.items():
            if currency not in merged_entries:
                merged_entries[currency] = amount
        schedules[schedule_id] = FixedFeeSchedule(
            entries=merged_entries,
            sources=existing.sources + sources,
            origin=existing.origin,
            inherited_from=existing.inherited_from,
            inheritance_reason=existing.inheritance_reason,
            inherited_sources=existing.inherited_sources,
        )
    else:
        schedules[schedule_id] = FixedFeeSchedule(entries=entries, sources=sources)


def _extract_international_surcharge_schedule(
    table: Table, base_name: str, source: Source | None = None
) -> dict[str, InternationalSurchargeSchedule]:
    """Extract international-surcharge schedules grouped by applicability signature."""
    table_text = _table_context_original(table)
    groups: dict[frozenset[tuple[str, Any]], list[InternationalSurchargeScheduleEntry]] = {}
    group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
    fallback: list[tuple[dict[str, Any], str, str]] = []
    for row in table.rows:
        label = _row_label(row)
        if _is_limit_or_cap_row(label, _row_fee_cell(row)):
            continue
        pct = _first_percentage(row)
        region = _normalize_region(label)
        if pct is None:
            # "No fee" / "Free" entries represent a 0% surcharge.
            fee_text = _norm(_row_fee_cell(row))
            no_fee_phrases = (
                "no fee",
                "free",
                "nessuna tariffa",
                "nessun costo",
                "ei palkkiota",
                "ei maksua",
                "bez poplatku",
                "bez poplatkov",
                "geen kosten",
                "ingen avgift",
                "ingen gebyr",
                "χωρίς χρέωση",
                "χωρισ χρεωση",
                "díjmentes",
                "nincs díj",
                "0%",
                "0,00%",
                "0.00%",
            )
            if any(phrase in fee_text for phrase in no_fee_phrases):
                pct = "0"
            else:
                continue
        signature = _schedule_signature_for_row(row, base_name, table_text, use_row_label=False)
        if region is None:
            # Some region-less tables (e.g. Brazil) list transaction types instead of
            # payer regions. Keep these as a fallback in case no region is recognized.
            if label:
                fallback.append((signature, label, pct))
            continue
        key = _signature_key(signature)
        group_keys.setdefault(key, signature)
        group_entries = groups.setdefault(key, [])
        # Avoid duplicate regions within the same applicability group.
        if any(e.payer_region == region for e in group_entries):
            continue
        group_entries.append(InternationalSurchargeScheduleEntry(payer_region=region, percentage_points=pct))
    if not groups and fallback:
        # No recognized region rows: treat the first percentage row as a generic
        # "OTHER" international surcharge. This is typically a region-less rate.
        signature, label, pct = fallback[0]
        key = _signature_key(signature)
        group_keys[key] = signature
        groups[key] = [InternationalSurchargeScheduleEntry(payer_region="OTHER", percentage_points=pct)]
    if not groups:
        return {}

    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    result: dict[str, InternationalSurchargeSchedule] = {}
    for key, entries in groups.items():
        suffix = _schedule_suffix_from_signature(group_keys[key])
        result[suffix] = InternationalSurchargeSchedule(entries=entries, sources=sources)
    return result


_REGION_EXACT: dict[str, str] = {
    "eu": "EEA",
    "gb": "GB",
    "uk": "GB",
    "us": "US_CA",
    "eøs": "EEA",
    "ees": "EEA",
    "eea": "EEA",
    "eee": "EEA",
    "ehp": "EEA",
    "egt": "EEA",
    "eta": "EEA",
    "see": "EEA",
    "eer": "EEA",
    "εοχ": "EEA",
}

# Substring patterns for each region, ordered by priority.
RegionPattern = str | tuple[str, ...]


_REGION_PATTERNS: tuple[tuple[str, tuple[RegionPattern, ...]], ...] = (
    ("EUROPE_II", ("europa ii",)),
    ("EUROPE_I", ("europa i",)),
    (
        "NORTHERN_EUROPE",
        ("nordeuropa", "northern europe", "nordic", "pohjois-eurooppa"),
    ),
    (
        "EEA",
        (
            "europäischer wirtschaftsraum",
            "ewr",
            "eea",
            "e.u",
            "eøs",
            "ees",
            "see",
            "eee",
            "ehp",
            "egt",
            "eta",
            "eer",
            "εοχ",
            "espace économique européen",
            "spazio economico europeo",
            "espacio económico europeo",
            "europæiske økonomiske samarbejdsområde",
            "europeisk økonomisk samarbeidsområde",
            "europeiska ekonomiska samarbetsområdet",
            "europese economische ruimte",
            "euroopan talousalue",
            "europski gospodarski prostor",
            "európai gazdasági térség",
        ),
    ),
    (
        "GB",
        (
            "vereinigtes königreich",
            "großbritannien",
            "storbritannien",
            "storbritannia",
            "united kingdom",
            "britain",
            "regno unito",
            "royaume-uni",
            "royaume uni",
            "verenigd koninkrijk",
            "iso-britannia",
            "Ηνωμένο Βασίλειο",
            "ηνωμενο βασιλειο",
            "britannien",
            "spojuené kráľovstvo",
            "spojené království",
            "egyesült királyság",
            "britannia",
        ),
    ),
    (
        "US_CA",
        (
            "usa",
            "united states",
            "u.s",
            "canada",
            "nordamerika",
            "états-unis",
            "etats-unis",
            "stati uniti",
            "estados unidos",
            "verenigde staten",
            "yhdysvallat",
            "ΗΠΑ",
            "ηπα",
        ),
    ),
    (
        "OTHER",
        (
            ("all", "other"),
            ("all", "andere"),
            ("tutti", "altri"),
            ("tous", "autres"),
            ("kaikki", "muut"),
            ("alle", "andere"),
            ("alle", "ander"),
            ("všechny", "ostatní"),
            ("všetky", "ostatné"),
            ("minden", "egyéb"),
            ("λοιπες",),
            ("rest",),
            ("restante",),
            ("andere",),
            ("sonstige",),
            ("welt",),
            ("andre", "markeder"),
            ("andre", "lande"),
            ("andere", "länder"),
            ("altri", "paesi"),
            ("altri", "mercati"),
            ("otros", "países"),
            ("otros", "mercados"),
            ("inni", "kraje"),
            ("pozostale",),
            ("pozostałe",),
            ("andre", "marknader"),
            ("alla", "andra", "marknader"),
            ("alle", "andere", "markten"),
            ("kaikki", "muut", "markkinat"),
            ("všechny", "ostatní", "trhy"),
            ("všetky", "ostatné", "trhy"),
            ("minden", "egyéb", "piac"),
            ("λοιπες", "αγορες"),
        ),
    ),
)


def _matches_region_pattern(text: str, pattern: RegionPattern) -> bool:
    if isinstance(pattern, str):
        return pattern in text
    return all(part in text for part in pattern)


def _normalize_region(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    if t in _REGION_EXACT:
        return _REGION_EXACT[t]
    for region, patterns in _REGION_PATTERNS:
        if any(_matches_region_pattern(t, p) for p in patterns):
            return region
    return None


# ---------------------------------------------------------------------------
# Reference detection and resolution
# ---------------------------------------------------------------------------

_REFERENCE_SCHEDULE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "online_card_payments": (
        "online-kartenzahlungen",
        "online card payments",
        "erweiterte kredit- und debitkartenzahlungen",
        "advanced credit and debit card",
        "online card",
    ),
    "alternative_payment_methods": (
        "alternative zahlungsmethode",
        "alternative payment methods",
        "alternative payment",
        "apm",
    ),
    "goods_and_services": (
        "waren und dienstleistungen",
        "goods and services",
    ),
    "donations": (
        "spenden",
        "donation",
    ),
    "nonprofit": (
        "gemeinnützig",
        "nonprofit",
    ),
    "other_commercial": (
        "geschäftliche transaktionen",
        "commercial transactions",
        "commercial transaction fees",
        "comercial",
        "transacciones comerciales",
        "transações comerciais",
        "transazioni commerciali",
        "transactions commerciales",
    ),
}

# Map a reference suffix to the product id it denotes.
_REFERENCE_SUFFIX_TO_PRODUCT: dict[str, str] = {
    "advanced": "advanced_card_payments",
}

# Inverse: map a product id to the reference suffix used in a qualified reference.
_REFERENCE_PRODUCT_SUFFIX: dict[str, str] = {v: k for k, v in _REFERENCE_SUFFIX_TO_PRODUCT.items()}


def _detect_reference(row: Row, product_id: str | None) -> str | None:
    """Detect when a row does not contain a numeric rate but refers to another schedule."""
    if _row_has_percentage(row):
        return None
    label_text = _norm(_row_label(row))
    fee_text = _norm(_row_fee_cell(row))
    if not fee_text or "{{" in fee_text:
        return None
    # A single-cell row with a reference-looking label is usually a section
    # header, not a textual schedule reference.
    non_empty_cells = [c for c in row.cells if c.text.strip()]
    if len(non_empty_cells) == 1 and fee_text == label_text:
        return None
    # A reference is a textual pointer; if it already contains money, it is
    # likely a flat-fee rule, not a reference.
    if _first_money(row):
        return None
    for schedule_name, keywords in _REFERENCE_SCHEDULE_KEYWORDS.items():
        for kw in keywords:
            if _norm(kw) in fee_text:
                suffix = _REFERENCE_PRODUCT_SUFFIX.get(product_id or "", "")
                if suffix:
                    return f"{schedule_name}.{suffix}"
                return schedule_name
    return None


def _reference_product_id(reference: str) -> str | None:
    """Return the product id a textual reference points to."""
    if "." in reference:
        base, suffix = reference.split(".", 1)
        return _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix, base)
    return reference


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _market_code_from_url(url: str | None) -> str | None:
    """Return the 2-letter market code from a PayPal URL path, if present."""
    if not url:
        return None
    match = re.search(r"paypal\.com/([a-zA-Z0-9]+)/", url)
    if match:
        return match.group(1).upper()
    return None


def _condition_score(rule: TransactionFeeRule, source_conditions: dict[str, Any]) -> int:
    """Score how specific a rule's conditions are relative to the source context.

    Higher scores mean the rule is a better match. Keys that are present in the
    source and match the rule are rewarded; extra, unrequested constraints in
    the rule are penalised. List-valued keys are also penalised by length (shorter
    lists are more specific), and the generic ``all_other_markets`` fallback is
    heavily penalised.
    """
    score = 0
    for key, rule_value in (rule.conditions or {}).items():
        if key in source_conditions:
            # A matching key is strong evidence that this rule is the right one.
            if key in ("applies_to_markets", "payment_methods"):
                if key == "applies_to_markets" and isinstance(rule_value, list) and "all_other_markets" in rule_value:
                    score -= 1000
                elif isinstance(rule_value, list):
                    score += 100 - len(rule_value)
                else:
                    score += 100
            else:
                score += 100
        else:
            # An extra, unrequested condition makes the rule too specific for a
            # generic source row.
            if key in ("applies_to_markets", "payment_methods"):
                penalty = 50
                if isinstance(rule_value, list):
                    penalty += len(rule_value)
                score -= penalty
            else:
                score -= 10
    return score


def _conditions_match_for_reference(
    rule_conditions: dict[str, Any],
    source_conditions: dict[str, Any],
) -> bool:
    """Return True when rule conditions are compatible with the source context.

    For ``applies_to_markets`` the source values must be included in the rule's
    list (the rule applies to the source market). For ``payment_methods`` the
    rule's methods must be a subset of the source's methods.  Scalar keys must
    match exactly.  When the source does not constrain a list key, the rule must
    not either (or use the generic ``all_other_markets`` fallback) so that a
    generic APM row does not resolve to a method-specific rule.
    """
    for key, rule_value in rule_conditions.items():
        source_value = source_conditions.get(key)
        if key == "applies_to_markets":
            if source_value is None or source_value == []:
                if _as_list(rule_value) == ["all_other_markets"]:
                    continue
                return False
            rule_markets = _as_list(rule_value)
            if not all(m in rule_markets for m in _as_list(source_value)):
                return False
        elif key == "payment_methods":
            if source_value is None or source_value == []:
                return False
            source_methods = _as_list(source_value)
            if not all(m in source_methods for m in _as_list(rule_value)):
                return False
        elif source_value is None:
            # The source has no constraint for this scalar key; only accept the
            # rule if its scalar value is the generic/default value, i.e. the
            # key is not actually narrowing the rule.
            continue
        elif rule_value != source_value:
            return False
    return True


def _reference_target_id(reference: str) -> str:
    """Resolve a textual reference to the target rule id it points to."""
    if "." not in reference:
        return reference
    base, suffix = reference.split(".", 1)
    suffix_product = _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix)
    return suffix_product or _REFERENCE_SUFFIX_TO_PRODUCT.get(base, base)


def _reference_candidates(rules: list[TransactionFeeRule | None], target_id: str) -> list[TransactionFeeRule]:
    """Find candidate rules matching a reference target id or label aliases."""
    candidates = [r for r in rules if r is not None and r.id == target_id and r.percentage is not None]
    if not candidates:
        aliases = _NORMALIZED_PRODUCT_ALIASES.get(target_id, ())
        candidates = [r for r in rules if r is not None and r.label and any(a in _norm(r.label) for a in aliases)]
    return candidates


def _resolved_source_conditions(
    candidates: list[TransactionFeeRule],
    source_conditions: dict[str, Any] | None,
    source: Provenance | None,
) -> dict[str, Any]:
    """Build source conditions, injecting the page market only when useful."""
    resolved = dict(source_conditions or {})
    if "applies_to_markets" not in resolved and source:
        market = _market_code_from_url(source.requested_url)
        if market and any(market in _as_list((r.conditions or {}).get("applies_to_markets")) for r in candidates):
            resolved["applies_to_markets"] = [market]
    return resolved


def _condition_matched_candidates(
    candidates: list[TransactionFeeRule],
    conditions: dict[str, Any],
) -> list[TransactionFeeRule]:
    """Filter candidates by condition compatibility and tie-break by specificity."""
    matched = [r for r in candidates if _conditions_match_for_reference(r.conditions or {}, conditions)]
    if matched and len(matched) > 1:
        max_score = max(_condition_score(r, conditions) for r in matched)
        matched = [r for r in matched if _condition_score(r, conditions) == max_score]
    return matched


def _build_resolved_rate(rule: TransactionFeeRule) -> ResolvedRate:
    return ResolvedRate(
        percentage=rule.percentage,
        fixed_fee_schedule=rule.fixed_fee_schedule,
        international_surcharge_schedule=rule.international_surcharge_schedule,
        maximum_fee_schedule=rule.maximum_fee_schedule,
        source=rule.source,
        rule_id=rule.id,
    )


def _resolve_reference(
    reference: str,
    rules: list[TransactionFeeRule | None],
    source_variant_id: str | None = None,
    source_conditions: dict[str, Any] | None = None,
    source: Provenance | None = None,
) -> tuple[ResolvedRate | None, bool]:
    """Resolve a textual reference to a concrete percentage and schedule names.

    A reference resolves only unambiguously. If more than one target rule
    matches, the reference is reported as ambiguous and ``(None, True)`` is
    returned. The source variant, source conditions and source provenance are
    used to disambiguate when the reference is tied to a specific variant or
    context.
    """
    target_id = _reference_target_id(reference)
    candidates = _reference_candidates(rules, target_id)
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return _build_resolved_rate(candidates[0]), False

    resolved_source_conditions = _resolved_source_conditions(candidates, source_conditions, source)

    matched = _condition_matched_candidates(candidates, resolved_source_conditions)
    if matched:
        if len(matched) == 1:
            return _build_resolved_rate(matched[0]), False
        candidates = matched

    if resolved_source_conditions and "transaction_region" in resolved_source_conditions:
        relaxed = _condition_matched_candidates(
            candidates,
            {k: v for k, v in resolved_source_conditions.items() if k != "transaction_region"},
        )
        if relaxed:
            if len(relaxed) == 1:
                return _build_resolved_rate(relaxed[0]), False
            candidates = relaxed

    if source_variant_id is not None:
        matched = [r for r in candidates if r.variant_id == source_variant_id]
        if len(matched) == 1:
            return _build_resolved_rate(matched[0]), False
        if not matched:
            default_candidates = [r for r in candidates if r.variant_id in (None, "default", "standard")]
            if len(default_candidates) == 1:
                return _build_resolved_rate(default_candidates[0]), False

    return None, True


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance(
    table: Table,
    row: Row,
    row_index: int,
    source: Source | None,
    section_heading: str | None = None,
    original_label: str | None = None,
) -> Provenance:
    return Provenance(
        requested_url=source.requested_url if source else None,
        canonical_url=source.canonical_url if source else None,
        page_id=source.page_id if source else None,
        page_title=source.page_title if source else None,
        document_id=row.source_document_id or table.document_id,
        component_id=row.source_component_id or table.component_id,
        table_id=table.table_id,
        row_id=row.row_id,
        row_index=row_index,
        section_heading=section_heading or (table.section_path[-1] if table.section_path else table.caption),
        original_label=original_label,
        classifier_version=_CLASSIFIER_VERSION,
    )


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def _parse_rate_expression(fee_text: str) -> tuple[str | None, str | None]:
    """Parse a German/English percentage + fixed-fee expression.

    Returns (percentage, fixed_fee_currency_amount_text).
    """
    pct: str | None = None
    # Find a percentage token anywhere in the text.
    for match in _PERCENTAGE_RE.finditer(fee_text):
        pct = normalize_decimal_string(match.group(1))
        break
    # Money amount is everything after the plus/extra token, if present.
    fixed: str | None = None
    plus_match = _PLUS_FIXED_RE.search(fee_text)
    if plus_match:
        fixed = plus_match.group(1).strip()
    return pct, fixed


@dataclass(frozen=True)
class _ExtractedRule:
    product_id: str
    variant_id: str | None
    label: str
    percentage: str | None
    fixed_fee_schedule: str | None
    international_surcharge_schedule: str | None
    maximum_fee_schedule: str | None
    conditions: dict[str, Any]
    table: Table
    row: Row
    row_index: int
    reference: str | None = None
    unknown_apm_methods: list[str] = field(default_factory=list)
    fee_components: list[FeeComponent] = field(default_factory=list)
    # Source schedule ids that an intended schedule may be inherited from when the
    # product-specific schedule is not directly present.
    fixed_fee_schedule_source: str | None = None
    international_surcharge_schedule_source: str | None = None
    maximum_fee_schedule_source: str | None = None
    fixed_expr: str | None = None
    table_category: str | None = None


def _classify_product_or_apm(label: str) -> tuple[str | None, list[str]]:
    """Classify a row label, treating APM special labels as unambiguous APM."""
    if _is_apm_special_label(label):
        return "alternative_payment_methods", []
    return _classify_product(label)


def _resolve_ambiguous_product(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    ambiguous_candidates: list[str],
    default_product: str | None,
    force_default_product: bool,
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> str | None:
    """Decide how to handle a row with ambiguous product candidates.

    Returns the resolved product id, or None if the row is queued for
    ``ambiguous`` / ``ignored``.
    """
    if force_default_product and default_product:
        return default_product
    if _row_has_percentage(row) or _has_likely_numeric_fee_candidate(row, table):
        ambiguous.append(
            AmbiguousFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                candidates=ambiguous_candidates,
            )
        )
        return None
    # A row with no determinable rate is informational, not a genuine ambiguity.
    ignored.append(
        UnclassifiedFeeRow(
            normalized_cells=_row_cells_text(row),
            original_label=label,
            source=_provenance(table, row, idx, source, original_label=label),
            reason="ambiguous product without fee",
        )
    )
    return None


def _resolve_missing_product(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    reference: str | None,
    default_product: str | None,
    force_default_product: bool,
    unclassified: list[UnclassifiedFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> str | None:
    """Resolve a product id for rows that did not match any product alias."""
    # Category-specific tables always fall back to their default product when
    # the label is not a product name.
    if force_default_product and default_product and (_row_has_percentage(row) or reference):
        return default_product
    # For mixed-product rate tables (e.g. commercial), use the default product
    # only when the row has its own rate and does not explicitly reference a
    # different product family.
    if default_product and not reference and _row_has_percentage(row):
        return default_product
    if reference:
        ref_product = _reference_product_id(reference)
        if ref_product:
            # A reference row that carries no product alias should be tagged
            # with the product it points to rather than the table's default.
            return ref_product
    if len(label) > 3 and _row_has_percentage(row):
        unclassified.append(
            UnclassifiedFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                reason="no product alias matched",
            )
        )
        return None
    if len(label) > 3 and _has_likely_numeric_fee_candidate(row, table):
        unclassified.append(
            UnclassifiedFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                reason="unclassified_fee_candidate",
            )
        )
        return None
    ignored.append(
        UnclassifiedFeeRow(
            normalized_cells=_row_cells_text(row),
            original_label=label,
            source=_provenance(table, row, idx, source, original_label=label),
            reason="no product alias and no rate",
        )
    )
    return None


def _resolve_product_id(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    default_product: str | None,
    force_default_product: bool,
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> tuple[str | None, str | None]:
    """Determine the product id and textual reference for a single table row."""
    product_id, ambiguous_candidates = _classify_product_or_apm(label)
    if ambiguous_candidates:
        product_id = _resolve_ambiguous_product(
            label,
            row,
            idx,
            table,
            source,
            ambiguous_candidates,
            default_product,
            force_default_product,
            ambiguous,
            ignored,
        )
        if product_id is None:
            return None, None
    if force_default_product and default_product:
        product_id = default_product

    reference = _detect_reference(row, product_id)
    if product_id is None:
        product_id = _resolve_missing_product(
            label,
            row,
            idx,
            table,
            source,
            reference,
            default_product,
            force_default_product,
            unclassified,
            ignored,
        )
        if product_id is None:
            return None, None
    return product_id, reference


def _ignored_rate_row(
    row: Row,
    row_index: int,
    label: str,
    table: Table,
    source: Source | None,
    reason: str,
) -> UnclassifiedFeeRow:
    """Return an ignored fee row with the given reason."""
    return UnclassifiedFeeRow(
        normalized_cells=_row_cells_text(row),
        original_label=label,
        source=_provenance(table, row, row_index, source, original_label=label),
        reason=reason,
    )


def _build_direct_fixed_rules(
    row: Row,
    row_index: int,
    product_id: str,
    fallback_variant_id: str,
    label: str,
    methods: list[str],
    table: Table,
    source: Source | None,
    direct_amounts: list[tuple[str, str, str]],
) -> list[_ExtractedRule]:
    """Build direct fixed-fee rules from a row's parsed amounts.

    When multiple currencies apply to the same variant, a ``fee_currency``
    condition is added so the rules have distinct identities.
    """
    rules: list[_ExtractedRule] = []
    variant_currencies: dict[str, set[str]] = {}
    for _, currency, amount_variant_id in direct_amounts:
        variant_currencies.setdefault(amount_variant_id, set()).add(currency)

    for amount, currency, amount_variant_id in direct_amounts:
        if amount_variant_id == "standard" and fallback_variant_id not in (None, "standard"):
            amount_variant_id = fallback_variant_id
        amount_conditions = _conditions_for_row(product_id, amount_variant_id, label, methods=methods, table=table)
        if len(variant_currencies.get(amount_variant_id, set())) > 1:
            amount_conditions["fee_currency"] = currency
        rules.append(
            _ExtractedRule(
                product_id=product_id,
                variant_id=amount_variant_id,
                label=label,
                percentage=None,
                fixed_fee_schedule=None,
                international_surcharge_schedule=None,
                maximum_fee_schedule=None,
                conditions=amount_conditions,
                table=table,
                row=row,
                row_index=row_index,
                reference=None,
                unknown_apm_methods=[],
                fee_components=[FeeComponent(type="fixed_amount", amount=amount, currency=currency)],
            )
        )
    return rules


def _build_standard_rate_rule(
    row: Row,
    row_index: int,
    product_id: str,
    variant_id: str,
    label: str,
    pct: str | None,
    reference: str | None,
    methods: list[str],
    unknown_methods: list[str],
    conditions: dict[str, Any],
    table: Table,
    table_category: str,
    fixed_expr: str | None,
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> _ExtractedRule:
    """Build a standard percentage/reference rule with schedule references."""
    fixed_schedule: str | None = None
    fixed_schedule_source: str | None = None
    if fixed_expr and product_id != "withdrawals":
        fixed_base = _fixed_fee_schedule_for(product_id, variant_id)
        if fixed_base:
            sig = _signature_from_conditions(conditions, fixed_base, product_id)
            fixed_schedule, fixed_schedule_source = _select_schedule_id(
                fixed_base,
                sig,
                fixed_schedules,
                _FIXED_FEE_SCHEDULE_FALLBACK.get(product_id, ()),
                product_base=_FIXED_FEE_SCHEDULE_FOR.get(product_id),
            )

    intl_schedule: str | None = None
    intl_schedule_source: str | None = None
    intl_base = _international_surcharge_schedule_for(product_id, variant_id)
    if intl_base:
        sig = _signature_from_conditions(conditions, intl_base, product_id)
        intl_schedule, intl_schedule_source = _select_schedule_id(
            intl_base,
            sig,
            international_schedules,
            _INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK.get(product_id, ()),
            product_base=_INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.get(product_id),
        )

    maximum_fee_schedule: str | None = None
    maximum_fee_schedule_source: str | None = None
    if product_id == "withdrawals" and table_category == "withdrawals_rate_table" and pct is not None:
        max_base = _maximum_fee_schedule_for_conditions(conditions)
        if max_base:
            sig = _signature_from_conditions(conditions, max_base, product_id)
            maximum_fee_schedule, maximum_fee_schedule_source = _select_schedule_id(
                max_base,
                sig,
                maximum_fee_schedules,
                _MAXIMUM_FEE_SCHEDULE_FALLBACK.get(max_base, ()),
                product_base=max_base,
            )

    # Listed-campaign donation campaigns are free.
    if variant_id == "campaign_unlisted":
        pct = "0"
        fixed_schedule = None
        intl_schedule = None

    return _ExtractedRule(
        product_id=product_id,
        variant_id=variant_id,
        label=label,
        percentage=pct,
        fixed_fee_schedule=fixed_schedule,
        international_surcharge_schedule=intl_schedule,
        maximum_fee_schedule=maximum_fee_schedule,
        conditions=conditions,
        table=table,
        row=row,
        row_index=row_index,
        reference=reference,
        unknown_apm_methods=unknown_methods,
        fixed_fee_schedule_source=fixed_schedule_source,
        international_surcharge_schedule_source=intl_schedule_source,
        maximum_fee_schedule_source=maximum_fee_schedule_source,
        fixed_expr=fixed_expr,
        table_category=table_category,
    )


def _handle_unusable_rate_row(
    row: Row,
    row_index: int,
    label: str,
    product_id: str,
    table: Table,
    source: Source | None,
    unclassified: list[UnclassifiedFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> None:
    """Store a row without a usable percentage/reference in the right bucket."""
    if _has_likely_numeric_fee_candidate(row, table):
        unclassified.append(_ignored_rate_row(row, row_index, label, table, source, "unsupported_fee_shape"))
    else:
        ignored.append(_ignored_rate_row(row, row_index, label, table, source, "no rate or reference"))


def _extract_rules_from_rate_table(
    table: Table,
    table_category: str,
    source: Source | None,
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> tuple[list[_ExtractedRule], list[UnclassifiedFeeRow], list[AmbiguousFeeRow], list[UnclassifiedFeeRow], int, int]:
    rules: list[_ExtractedRule] = []
    unclassified: list[UnclassifiedFeeRow] = []
    ambiguous: list[AmbiguousFeeRow] = []
    ignored: list[UnclassifiedFeeRow] = []
    numeric_fee_candidates = 0
    unclassified_fee_candidates = 0

    default_product = _TABLE_CATEGORY_PRODUCT.get(table_category)
    force_default_product = table_category in _CATEGORY_SPECIFIC_TABLES

    for idx, row in enumerate(table.rows):
        label = _row_label(row)
        fee_text = _row_fee_cell(row)
        is_limit_or_cap = _is_limit_or_cap_row(label, fee_text)
        if _has_likely_numeric_fee_candidate(row, table) and not is_limit_or_cap:
            numeric_fee_candidates += 1
        if not label:
            ignored.append(_ignored_rate_row(row, idx, label, table, source, "empty label"))
            continue

        product_id, reference = _resolve_product_id(
            label,
            row,
            idx,
            table,
            source,
            default_product,
            force_default_product,
            unclassified,
            ambiguous,
            ignored,
        )
        if product_id is None:
            continue

        if is_limit_or_cap:
            ignored.append(_ignored_rate_row(row, idx, label, table, source, "limit or cap"))
            continue

        pct, fixed_expr = _parse_rate_expression(fee_text)
        methods, unknown_methods = _extract_apm_methods(label)
        variant_id = _variant_id_for_row(product_id, label, methods, table, fee_text=fee_text) or "standard"

        if product_id == "alternative_payment_methods" and variant_id in _APM_VARIANT_ONLY_VARIANTS:
            unknown_methods = []

        if pct is None and reference is None and product_id in _DIRECT_FIXED_FEE_PRODUCTS:
            direct_amounts = _extract_direct_fixed_amounts(row, product_id, table, source)
            if direct_amounts:
                rules.extend(
                    _build_direct_fixed_rules(
                        row,
                        idx,
                        product_id,
                        variant_id,
                        label,
                        methods,
                        table,
                        source,
                        direct_amounts,
                    )
                )
                continue

        conditions = _conditions_for_row(product_id, variant_id, label, methods=methods, table=table)

        if pct is None and reference is None:
            _handle_unusable_rate_row(row, idx, label, product_id, table, source, unclassified, ignored)
            continue

        rules.append(
            _build_standard_rate_rule(
                row,
                idx,
                product_id,
                variant_id,
                label,
                pct,
                reference,
                methods,
                unknown_methods,
                conditions,
                table,
                table_category,
                fixed_expr,
                fixed_schedules,
                international_schedules,
                maximum_fee_schedules,
            )
        )
    unclassified_fee_candidates = sum(
        1 for r in unclassified if r.reason in ("unclassified_fee_candidate", "unsupported_fee_shape")
    )
    return rules, unclassified, ambiguous, ignored, numeric_fee_candidates, unclassified_fee_candidates


# Maps a product to its own fixed-fee schedule. The target schedule may be a
# product-specific schedule (e.g. goods_and_services) or None if the product has
# no fixed fee. When a product-specific schedule is missing from the extracted
# schedules, _resolve_schedule_inheritance can create an explicitly inherited
# schedule so the rule remains resolvable without merging into a generic
# commercial schedule.
_FIXED_FEE_SCHEDULE_FOR: dict[str, str | None] = {
    "paypal_checkout": "paypal_checkout",
    "goods_and_services": "goods_and_services",
    "online_card_payments": "online_card_payments",
    "advanced_card_payments": "advanced_card_payments",
    "other_commercial": "other_commercial",
    "guest_checkout": "guest_checkout",
    "invoice_pay_later": "invoice_pay_later",
    "pay_later_consumer": "pay_later_consumer",
    "qr_code_payments": "qr_code_payments",
    "donations": "donations",
    "nonprofit": "nonprofit",
    "micropayments": "micropayments",
    "alternative_payment_methods": "alternative_payment_methods",
    "pos_transactions": None,
    "chargebacks": "chargebacks",
    "refunds": "refunds",
    "disputes": "disputes",
    "card_verification": "card_verification",
    "currency_conversion": None,
    "withdrawals": "withdrawals",
}

# Subset of _FIXED_FEE_SCHEDULE_FOR that represents explicit inheritance.
# When a product is listed here, its product-specific schedule may be inherited
# from the named source schedule family when the source schedule exists and the
# inheritance is documented by an explicit product rule or source text.
_FIXED_FEE_INHERITANCE: dict[str, str] = {
    "paypal_checkout": "commercial",
    "other_commercial": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "advanced_card_payments": "online_card_payments",
}


def _fixed_fee_schedule_for(product_id: str, variant_id: str | None = None) -> str | None:
    """Return the fixed-fee schedule name for a product and variant, or None if no fixed fee applies."""
    base = _FIXED_FEE_SCHEDULE_FOR.get(product_id)
    if base is None:
        return None
    if variant_id is None or variant_id == "standard":
        return base
    if variant_id in _BASE_VARIANTS_BY_PRODUCT.get(product_id, frozenset()):
        return base
    return f"{base}_{variant_id}"


def _international_surcharge_schedule_for(product_id: str, variant_id: str | None = None) -> str | None:
    """Return the international surcharge schedule name for a product and variant, or None."""
    base = _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.get(product_id)
    if base is None:
        return None
    if variant_id is None or variant_id == "standard":
        return base
    if variant_id in _BASE_VARIANTS_BY_PRODUCT.get(product_id, frozenset()):
        return base
    return f"{base}_{variant_id}"


# Fallback schedule order per product used when the product-specific schedule
# is not present in the extracted data. Fallbacks are restricted to the
# explicitly declared inheritance relationships in _FIXED_FEE_INHERITANCE; there
# is no implicit fallback to a different schedule family.
_FIXED_FEE_SCHEDULE_FALLBACK: dict[str, tuple[str, ...]] = {
    "paypal_checkout": ("commercial",),
    "other_commercial": ("commercial",),
    "guest_checkout": ("commercial",),
    "invoice_pay_later": ("commercial",),
    "pay_later_consumer": ("commercial",),
    "advanced_card_payments": ("online_card_payments",),
}


# Same as above for international surcharge schedules.
_INTERNATIONAL_SURCHARGE_SCHEDULE_FOR: dict[str, str | None] = {
    "paypal_checkout": "paypal_checkout",
    "goods_and_services": "goods_and_services",
    "online_card_payments": "online_card_payments",
    "advanced_card_payments": "advanced_card_payments",
    "other_commercial": "other_commercial",
    "guest_checkout": "guest_checkout",
    "invoice_pay_later": "invoice_pay_later",
    "pay_later_consumer": "pay_later_consumer",
    "qr_code_payments": None,
    "donations": "donations",
    "nonprofit": "nonprofit",
    "micropayments": None,
    "alternative_payment_methods": None,
    "pos_transactions": None,
    "chargebacks": None,
    "refunds": None,
    "disputes": None,
    "card_verification": None,
    "currency_conversion": None,
    "withdrawals": None,
}

_INTERNATIONAL_SURCHARGE_INHERITANCE: dict[str, str] = {
    "paypal_checkout": "commercial",
    "other_commercial": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "advanced_card_payments": "commercial",
}


# Fallback order for product-specific international surcharge schedules.
_INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK: dict[str, tuple[str, ...]] = {
    "paypal_checkout": ("commercial",),
    "other_commercial": ("commercial",),
    "guest_checkout": ("commercial",),
    "invoice_pay_later": ("commercial",),
    "pay_later_consumer": ("commercial",),
    "advanced_card_payments": ("commercial",),
}


# Fallback order for product-specific maximum-fee schedules.
_MAXIMUM_FEE_SCHEDULE_FALLBACK: dict[str, tuple[str, ...]] = {
    "payouts_us": ("payouts_international",),
}


# ---------------------------------------------------------------------------
# Schedule assembly
# ---------------------------------------------------------------------------


_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS: dict[str, str] = {
    "chargebacks": "chargebacks",
    "refunds": "refunds",
    "disputes": "disputes",
    "card_verification": "card_verification",
    "withdrawals": "withdrawals",
}


def _direct_fixed_product_variant(schedule_id: str, direct_bases: set[str]) -> tuple[str | None, str | None]:
    """Return the product id and variant id for a direct fixed-fee schedule id."""
    base_part = schedule_id.split("__", 1)[0]
    if base_part in direct_bases:
        return base_part, "standard"
    for base in sorted(direct_bases, key=len, reverse=True):
        if base_part.startswith(base + "_"):
            variant = base_part[len(base) + 1 :]
            return base, variant
    return None, None


def _schedule_ids_for_table(
    base_name: str,
    applicable_variants: list[str],
    existing_names: set[str],
    product_is_direct: bool = False,
) -> list[str]:
    """Determine schedule ids to create for a fixed/international surcharge table.

    If the table is generic (no variants) or describes a base variant, the
    base name is used. Otherwise variant-specific ids are created. Direct
    products (withdrawals, disputes) never create a base schedule when a
    variant is present.
    """
    if not applicable_variants:
        return [base_name]
    base_variants = _BASE_VARIANTS_BY_PRODUCT.get(base_name, frozenset())
    has_base_variant = bool(set(applicable_variants) & base_variants)
    if has_base_variant and base_name not in existing_names and not product_is_direct:
        return [base_name]
    return [f"{base_name}_{variant}" for variant in applicable_variants]


def _merge_fixed_like_schedules(
    existing: FixedFeeSchedule,
    schedule: FixedFeeSchedule,
    name: str,
    schedule_type: str,
) -> tuple[FixedFeeSchedule, list[Diagnostic]]:
    """Merge a fixed-like schedule into an existing one, reporting conflicts."""
    merged_entries = dict(existing.entries)
    merged_sources = list(existing.sources)
    diagnostics: list[Diagnostic] = []
    for s in schedule.sources:
        if s not in merged_sources:
            merged_sources.append(s)
    for currency, amount in schedule.entries.items():
        if currency in merged_entries:
            if merged_entries[currency] != amount:
                diagnostics.append(
                    Diagnostic(
                        type="conflicting_schedule_entry",
                        schedule_type=schedule_type,
                        schedule_id=name,
                        normalized_key=currency,
                        values=[merged_entries[currency], amount],
                        sources=_merge_provenance_sources(existing.sources, schedule.sources),
                    )
                )
            # Keep first value; do not overwrite.
        else:
            merged_entries[currency] = amount

    # Preserve inherited provenance; direct + direct stays direct.
    merged_inherited_sources = list(existing.inherited_sources)
    for s in schedule.inherited_sources:
        if s not in merged_inherited_sources:
            merged_inherited_sources.append(s)
    origin = existing.origin
    inherited_from = existing.inherited_from
    inheritance_reason = existing.inheritance_reason
    if schedule.origin == "inherited":
        origin = "inherited"
        inherited_from = schedule.inherited_from or inherited_from
        inheritance_reason = schedule.inheritance_reason or inheritance_reason
        for s in schedule.sources:
            if s not in merged_inherited_sources:
                merged_inherited_sources.append(s)
    return FixedFeeSchedule(
        entries=merged_entries,
        sources=merged_sources,
        origin=origin,
        inherited_from=inherited_from,
        inheritance_reason=inheritance_reason,
        inherited_sources=merged_inherited_sources,
    ), diagnostics


def _merge_international_surcharge_schedules(
    existing: InternationalSurchargeSchedule,
    schedule: InternationalSurchargeSchedule,
    name: str,
) -> tuple[InternationalSurchargeSchedule, list[Diagnostic]]:
    """Merge an international surcharge schedule into an existing one."""
    merged_entries = list(existing.entries)
    seen = {e.payer_region: e for e in merged_entries}
    merged_sources = list(existing.sources)
    diagnostics: list[Diagnostic] = []
    for s in schedule.sources:
        if s not in merged_sources:
            merged_sources.append(s)
    for entry in schedule.entries:
        if entry.payer_region in seen:
            if seen[entry.payer_region].percentage_points != entry.percentage_points:
                diagnostics.append(
                    Diagnostic(
                        type="conflicting_schedule_entry",
                        schedule_type="international_surcharge",
                        schedule_id=name,
                        normalized_key=entry.payer_region,
                        values=[
                            seen[entry.payer_region].percentage_points or "",
                            entry.percentage_points or "",
                        ],
                        sources=_merge_provenance_sources(existing.sources, schedule.sources),
                    )
                )
            # Keep first value; do not overwrite.
        else:
            merged_entries.append(entry)
            seen[entry.payer_region] = entry

    merged_inherited_sources = list(existing.inherited_sources)
    for s in schedule.inherited_sources:
        if s not in merged_inherited_sources:
            merged_inherited_sources.append(s)
    origin = existing.origin
    inherited_from = existing.inherited_from
    inheritance_reason = existing.inheritance_reason
    if schedule.origin == "inherited":
        origin = "inherited"
        inherited_from = schedule.inherited_from or inherited_from
        inheritance_reason = schedule.inheritance_reason or inheritance_reason
        for s in schedule.sources:
            if s not in merged_inherited_sources:
                merged_inherited_sources.append(s)
    return InternationalSurchargeSchedule(
        entries=merged_entries,
        sources=merged_sources,
        origin=origin,
        inherited_from=inherited_from,
        inheritance_reason=inheritance_reason,
        inherited_sources=merged_inherited_sources,
    ), diagnostics


def _collect_fixed_fee_table(
    table: Table,
    source: Source | None,
    fixed: dict[str, FixedFeeSchedule],
    direct_products: set[str],
) -> list[Diagnostic]:
    """Collect fixed-fee schedules from a single table into ``fixed``."""
    base_name = _schedule_name_from_table(table, "commercial")
    schedules_by_sig = _extract_fixed_fee_schedule(table, base_name, source=source)
    if not schedules_by_sig:
        return []
    applicable_variants = _applicable_variants_for_table(table, base_name)
    product_is_direct = base_name in direct_products
    diagnostics: list[Diagnostic] = []
    for base_id in _schedule_ids_for_table(base_name, applicable_variants, set(fixed.keys()), product_is_direct):
        for suffix, schedule in schedules_by_sig.items():
            name = f"{base_id}__{suffix}" if suffix else base_id
            existing = fixed.get(name)
            if existing:
                merged, new_diagnostics = _merge_fixed_like_schedules(existing, schedule, name, "fixed_fee")
                diagnostics.extend(new_diagnostics)
                fixed[name] = merged
            else:
                fixed[name] = schedule
    return diagnostics


def _collect_international_surcharge_table(
    table: Table,
    source: Source | None,
    international: dict[str, InternationalSurchargeSchedule],
) -> list[Diagnostic]:
    """Collect international-surcharge schedules from a single table."""
    base_name = _schedule_name_from_table(table, "commercial")
    schedules_by_sig = _extract_international_surcharge_schedule(table, base_name, source=source)
    if not schedules_by_sig:
        return []
    applicable_variants = _applicable_variants_for_table(table, base_name)
    diagnostics: list[Diagnostic] = []
    for base_id in _schedule_ids_for_table(base_name, applicable_variants, set(international.keys())):
        for suffix, schedule in schedules_by_sig.items():
            name = f"{base_id}__{suffix}" if suffix else base_id
            existing = international.get(name)
            if existing:
                merged, new_diagnostics = _merge_international_surcharge_schedules(existing, schedule, name)
                diagnostics.extend(new_diagnostics)
                international[name] = merged
            else:
                international[name] = schedule
    return diagnostics


def _collect_maximum_fee_table(
    table: Table,
    source: Source | None,
    maximum: dict[str, FixedFeeSchedule],
) -> list[Diagnostic]:
    """Collect maximum-fee schedules from a single table."""
    diagnostics: list[Diagnostic] = []
    for name, schedule in _extract_maximum_fee_schedule(table, source=source).items():
        existing = maximum.get(name)
        if existing:
            merged, new_diagnostics = _merge_fixed_like_schedules(existing, schedule, name, "maximum_fee")
            diagnostics.extend(new_diagnostics)
            maximum[name] = merged
        else:
            maximum[name] = schedule
    return diagnostics


def _collect_schedules(
    tables: list[Table],
    source: Source | None = None,
    table_categories: dict[int, str | None] | None = None,
) -> tuple[
    dict[str, FixedFeeSchedule],
    dict[str, InternationalSurchargeSchedule],
    dict[str, FixedFeeSchedule],
    list[Diagnostic],
]:
    """Extract fixed-fee, international-surcharge and maximum-fee schedules.

    Schedules are keyed by product name. If two tables map to the same product
    (e.g. "Fixed fee by received currency" and "Currency fixed fees" both for
    commercial), their entries are merged and sources are combined. Conflicting
    duplicate keys are reported as diagnostics and the first encountered value
    is kept.
    """
    fixed: dict[str, FixedFeeSchedule] = {}
    international: dict[str, InternationalSurchargeSchedule] = {}
    maximum: dict[str, FixedFeeSchedule] = {}
    diagnostics: list[Diagnostic] = []

    direct_products = set(_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS.values())

    for table in tables:
        category = table_categories[id(table)] if table_categories else _classify_table_category(table)
        if category == "fixed_fee_table":
            diagnostics.extend(_collect_fixed_fee_table(table, source, fixed, direct_products))
        elif category == "international_surcharge_table":
            diagnostics.extend(_collect_international_surcharge_table(table, source, international))
        elif category == "maximum_fee_table":
            diagnostics.extend(_collect_maximum_fee_table(table, source, maximum))

    return fixed, international, maximum, diagnostics


def _create_direct_fixed_fee_rules(
    fixed_schedules: dict[str, FixedFeeSchedule],
    referenced_schedules: set[str],
) -> list[TransactionFeeRule]:
    """Create calculable rules for schedules that are not referenced by a rate table.

    Chargebacks, disputes, refunds, card verification and withdrawals are
    expressed as direct monetary amounts per currency. They do not have a
    percentage-based rate table, so they cannot be represented by the normal
    rate-table extraction path.
    """
    rules: list[TransactionFeeRule] = []
    direct_bases = set(_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS.values())
    for schedule_id, schedule in fixed_schedules.items():
        if schedule_id in referenced_schedules:
            continue
        base_part = schedule_id.split("__", 1)[0]
        product_id, variant_id = _direct_fixed_product_variant(base_part, direct_bases)
        if not product_id:
            continue
        if not schedule.entries:
            continue
        provenance = schedule.sources[0] if schedule.sources else None
        conditions = _conditions_from_schedule_id(schedule_id)
        if product_id == "withdrawals" and variant_id and variant_id != "standard":
            conditions["withdrawal_method"] = variant_id
        elif product_id == "disputes" and variant_id and variant_id != "standard":
            conditions["volume_status"] = variant_id
        rules.append(
            TransactionFeeRule(
                id=product_id,
                variant_id=variant_id or "standard",
                label=provenance.section_heading if provenance else None,
                percentage=None,
                fixed_fee_schedule=schedule_id,
                international_surcharge_schedule=None,
                conditions=conditions,
                rate_reference=None,
                source=provenance,
                calculation_status="calculable",
                fee_components=[],
            )
        )
    return rules


def _source_schedule_id(source_base: str, intended_id: str, schedules: dict[str, Any]) -> str | None:
    """Choose the most specific source schedule matching the intended id's suffix."""
    if "__" in intended_id:
        suffix = intended_id.split("__", 1)[1]
        suffixed = f"{source_base}__{suffix}"
        if suffixed in schedules:
            return suffixed
    return source_base if source_base in schedules else None


def _inheritance_evidence(
    rule: TransactionFeeRule,
    extracted: _ExtractedRule | None,
    source_base: str,
    source_schedule_id: str,
    schedule_type: str,
    schedules: dict[str, Any],
) -> str | None:
    """Return a human-readable evidence string when inheritance is allowed."""
    if schedule_type == "fixed_fee":
        inheritance_map = _FIXED_FEE_INHERITANCE
    elif schedule_type == "international_surcharge":
        inheritance_map = _INTERNATIONAL_SURCHARGE_INHERITANCE
    else:
        inheritance_map = {}

    # The inheritance map is itself an explicit documented product rule.  Even
    # for mapped products we still require corroborating source text or table
    # context so there is no implicit cross-product fallback.
    mapped_source = inheritance_map.get(rule.id)
    if mapped_source and mapped_source != source_base:
        return None

    # Maximum-fee schedules have their own explicit fallback map and do not
    # require source-text corroboration.
    if schedule_type == "maximum_fee" and source_base:
        return f"explicit maximum-fee fallback from {source_base}"

    # Source text from the requesting row can explicitly name the source schedule.
    source_text = (extracted.fixed_expr or "").lower() if extracted else ""
    if source_text:
        if (
            source_base == "commercial"
            and "commercial" in source_text
            and ("transaction" in source_text or "fixed fee" in source_text)
        ):
            return "source text references commercial fixed fee"
        if source_base == "online_card_payments" and "online card" in source_text:
            return "source text references online card fixed fee"

    # Table context from the source schedule or the requesting row can
    # document inheritance (e.g. an Advanced Card row in an Online Card
    # Payment Services section).
    if source_base == "online_card_payments":
        if "online card" in source_text:
            return "table context references online card fixed fee"
        source_schedule = schedules.get(source_schedule_id)
        if source_schedule and source_schedule.sources:
            heading = (source_schedule.sources[0].section_heading or "").lower()
            if "online card" in heading:
                return "table context references online card fixed fee"
    if source_base == "commercial":
        if "commercial" in source_text and ("transaction" in source_text or "fixed fee" in source_text):
            return "source text references commercial fixed fee"
        source_schedule = schedules.get(source_schedule_id)
        if source_schedule and source_schedule.sources:
            heading = (source_schedule.sources[0].section_heading or "").lower()
            if "commercial transactions" in heading:
                return "table context references commercial fixed fee"

    # When the product is explicitly mapped and no contradictory evidence
    # exists, the map itself documents the inheritance.
    if mapped_source == source_base:
        return f"explicit product inheritance from {source_base}"

    return None


def _create_inherited_schedule(
    schedule_id: str,
    source_schedule_id: str,
    schedules: dict[str, Any],
    rule_source: Provenance | None,
    reason: str,
) -> None:
    source = schedules[source_schedule_id]
    inherited_sources = list(source.sources)
    sources = list(source.sources)
    if rule_source and rule_source not in sources:
        sources.append(rule_source)
    schedules[schedule_id] = source.model_copy(
        update={
            "origin": "inherited",
            "inherited_from": source_schedule_id,
            "inheritance_reason": reason,
            "inherited_sources": inherited_sources,
            "sources": sources,
        }
    )


def _resolve_schedule(
    schedule_id: str,
    source_id: str | None,
    schedule_type: str,
    schedules: dict[str, Any],
    inheritance_map: dict[str, str],
    rule: TransactionFeeRule,
    extracted: _ExtractedRule | None,
    diagnostics: list[Diagnostic],
) -> str | None:
    """Return the actual source schedule id if inheritance is allowed, else None."""
    if not source_id:
        return None
    source_base = source_id.split("__", 1)[0]
    expected_source = inheritance_map.get(rule.id)
    if expected_source and source_base != expected_source:
        return None
    actual_source = _source_schedule_id(source_base, schedule_id, schedules)
    if not actual_source:
        return None
    reason = _inheritance_evidence(rule, extracted, source_base, actual_source, schedule_type, schedules)
    if not reason:
        return None
    _create_inherited_schedule(schedule_id, actual_source, schedules, rule.source, reason)
    diagnostics.append(
        Diagnostic(
            type="inherited_schedule",
            rule_id=rule.id,
            schedule_type=schedule_type,
            expected_schedule=schedule_id,
            inherited_from=actual_source,
            sources=[rule.source] if rule.source else [],
        )
    )
    return actual_source


def _resolve_schedule_inheritance(
    extracted_rules: list[_ExtractedRule],
    rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Create inherited schedules for product rules whose own schedule is missing.

    Inheritance is only performed when the source text/table context contains an
    explicit reference to the source schedule family, or when the product is in
    the explicit inheritance map and the source text/table context supports it.
    Every inherited schedule records its source schedule and provenance from
    both the requesting rule and the source schedule.
    """

    _schedule_attrs: list[tuple[str, dict[str, Any], dict[str, str]]] = [
        ("fixed_fee", fixed_schedules, _FIXED_FEE_INHERITANCE),
        ("international_surcharge", international_schedules, _INTERNATIONAL_SURCHARGE_INHERITANCE),
        ("maximum_fee", maximum_fee_schedules, {}),
    ]

    # Collect all missing schedule references and create inherited schedules
    # before updating any rule.  Base schedules are created first (shorter ids)
    # so variant-specific ids can reuse an inherited base schedule.
    refs: list[
        tuple[str, str | None, str, dict[str, Any], dict[str, str], TransactionFeeRule, _ExtractedRule | None]
    ] = []
    for i, rule in enumerate(rules):
        extracted = extracted_rules[i] if i < len(extracted_rules) else None
        for schedule_type, schedules, inheritance_map in _schedule_attrs:
            schedule_id = getattr(rule, f"{schedule_type}_schedule")
            source_id = getattr(extracted, f"{schedule_type}_schedule_source", None)
            if schedule_id and schedule_id not in schedules:
                refs.append((schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted))

    # Deduplicate by (schedule_type, schedule_id), preferring an entry with a
    # concrete source id so we can resolve the source schedule.
    seen: dict[
        tuple[str, str],
        tuple[str, str | None, str, dict[str, Any], dict[str, str], TransactionFeeRule, _ExtractedRule | None],
    ] = {}
    for ref in refs:
        key = (ref[2], ref[0])
        existing = seen.get(key)
        if existing is None or (existing[1] is None and ref[1] is not None):
            seen[key] = ref
    unique_refs = sorted(seen.values(), key=lambda r: (len(r[0]), r[0]))

    created: set[tuple[str, str]] = set()
    schedule_source: dict[tuple[str, str], str | None] = {}
    for schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted in unique_refs:
        source = _resolve_schedule(
            schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted, diagnostics
        )
        if source:
            created.add((schedule_type, schedule_id))
        schedule_source[(schedule_type, schedule_id)] = source

    # Update every rule to use an inherited schedule when one was created, or
    # report it as missing.
    for i, rule in enumerate(rules):
        extracted = extracted_rules[i] if i < len(extracted_rules) else None
        updates: dict[str, Any] = {}
        for schedule_type, schedules, _ in _schedule_attrs:
            attr = f"{schedule_type}_schedule"
            schedule_id = getattr(rule, attr)
            if not schedule_id:
                continue
            if schedule_id in schedules:
                continue
            if (schedule_type, schedule_id) in created:
                continue
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type=schedule_type,
                    expected_schedule=schedule_id,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates[attr] = None
        if updates:
            rules[i] = rule.model_copy(update=updates)


def _product_family_for_schedule_id(schedule_id: str, schedule_type: str = "fixed_fee") -> str:
    """Return the product-family base name for a schedule id.

    For fixed-fee and international-surcharge schedules the family is the
    longest known product base that matches the id's prefix (e.g.
    ``micropayments_domestic`` -> ``micropayments``).  Maximum-fee schedules
    have no variant prefix, so the family is the base id before any ``__``
    applicability suffix.
    """
    base = schedule_id.split("__", 1)[0]
    if schedule_type == "maximum_fee":
        return base
    product_bases: set[str] = set()
    for value in _FIXED_FEE_SCHEDULE_FOR.values():
        if value:
            product_bases.add(value)
    for value in _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.values():
        if value:
            product_bases.add(value)
    candidates = [pb for pb in product_bases if base == pb or base.startswith(pb + "_")]
    if candidates:
        return max(candidates, key=len)
    return base


def _validate_inheritance_priorities(
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Flag inherited schedules that selected a cross-product fallback while a
    direct product-family schedule was available.

    The validation mirrors the resolution priority enforced by
    ``_select_schedule_id``: a variant-specific schedule may fall back to its
    own product base, but it must never fall back to a different product family
    when its own product base exists.
    """
    schedule_groups: list[tuple[str, dict[str, Any]]] = [
        ("fixed_fee", fixed_schedules),
        ("international_surcharge", international_schedules),
        ("maximum_fee", maximum_fee_schedules),
    ]
    for schedule_type, schedules in schedule_groups:
        invalid: list[str] = []
        for schedule_id, schedule in schedules.items():
            if schedule.origin != "inherited":
                continue
            target_family = _product_family_for_schedule_id(schedule_id, schedule_type)
            source_id = schedule.inherited_from or ""
            source_family = _product_family_for_schedule_id(source_id, schedule_type)
            if target_family == source_family:
                continue
            # Product-base schedules (possibly with an applicability suffix) may
            # be explicitly inherited from another product family; only
            # variant-specific schedules are checked for bypassing their own base.
            base_part = schedule_id.split("__", 1)[0]
            if base_part == target_family:
                continue
            suffix = ""
            if "__" in schedule_id:
                suffix = schedule_id.split("__", 1)[1]
            direct_candidates: list[str] = []
            if suffix:
                direct_candidates.append(f"{target_family}__{suffix}")
            direct_candidates.append(target_family)
            for direct in direct_candidates:
                if direct in schedules:
                    diagnostics.append(
                        Diagnostic(
                            type="inappropriate_inheritance",
                            rule_id=target_family,
                            schedule_type=schedule_type,
                            schedule_id=schedule_id,
                            inherited_from=source_id,
                            expected_schedule=direct,
                            sources=schedule.sources,
                        )
                    )
                    invalid.append(schedule_id)
                    break
        for schedule_id in invalid:
            del schedules[schedule_id]


def _merge_provenance_sources(*source_lists: list[Provenance]) -> list[Provenance]:
    """Combine multiple provenance lists without duplicates."""
    merged: list[Provenance] = []
    for sources in source_lists:
        for s in sources:
            if s not in merged:
                merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


_STATUS_DEFECT_DIAGNOSTICS: set[str] = {
    "missing_required_schedule",
    "conflicting_schedule_entry",
    "conflicting_rule_identity",
    "unresolved_reference",
    "unresolved_nested_reference",
    "ambiguous_identity",
    "unsupported_fee_shape",
    "ambiguous_reference",
    "unknown_apm_method",
    "inappropriate_inheritance",
}


def _derive_status(
    rules: list[TransactionFeeRule],
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
    diagnostics: list[Diagnostic],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> str:
    if not rules and not fixed_schedules and not international_schedules and not maximum_fee_schedules:
        return "unclassified"
    # Informational or explicitly ignored rows are not defects. Ambiguous and
    # unclassified rows are.
    if ambiguous or unclassified:
        return "partial"
    if any(d.type in _STATUS_DEFECT_DIAGNOSTICS for d in diagnostics):
        return "partial"
    # A complete result must expose at least one of the core PayPal payment
    # products (PayPal Checkout or goods and services).  The generic
    # ``other_commercial`` fallback alone does not imply full product coverage.
    required_core_ids = {"paypal_checkout", "goods_and_services"}
    generic_core_id = "other_commercial"
    core_ids = required_core_ids | {generic_core_id}
    core_rules = [r for r in rules if r.id in core_ids]
    found_required = any(r.id in required_core_ids for r in rules)
    if (
        found_required
        and core_rules
        and all(r.calculation_status == "calculable" for r in core_rules)
        and bool(fixed_schedules)
    ):
        return "complete"
    if any(r.id in required_core_ids for r in rules):
        return "partial"
    # Non-commercial markets may still be partial if any rule is incomplete.
    if any(r.calculation_status != "calculable" for r in rules if r.id not in core_ids):
        return "partial"
    return "partial"


def _rule_identity(rule: TransactionFeeRule) -> str:
    """Return a stable selector key for deduplicating equivalent rules.

    A rule selector is uniquely defined by product family, variant and
    applicable conditions. The localized ``label``, percentage and schedule
    references are intentionally excluded: the same selector with different
    fees is a conflict, not a different rule. Different rates must be
    expressed through different variants or conditions.
    """
    return json.dumps(
        {
            "id": rule.id,
            "variant_id": rule.variant_id,
            "conditions": {k: rule.conditions[k] for k in sorted(rule.conditions)} if rule.conditions else {},
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _rule_has_rate(rule: TransactionFeeRule) -> bool:
    """Return True if the rule carries a directly usable percentage."""
    return bool(
        rule.percentage is not None
        or (rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None)
    )


def _is_reference_source(rule: TransactionFeeRule) -> bool:
    """Return True if the rule is a reference that resolves to another rule."""
    return bool(rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None)


def _fee_components_for_rule(rule: TransactionFeeRule) -> list[FeeComponent]:
    """Build the explicit fee components for a transaction rule.

    The legacy ``percentage`` / ``fixed_fee_schedule`` / ``international_surcharge_schedule``
    fields are translated into a list of typed components so that a fee
    calculator can consume a single structure. Direct ``fixed_amount`` components
    already present on the rule are preserved.
    """
    components: list[FeeComponent] = []
    for component in rule.fee_components or []:
        if component.type == "fixed_amount":
            components.append(component)
    if rule.percentage is not None:
        components.append(FeeComponent(type="percentage", value=rule.percentage))
    if rule.fixed_fee_schedule is not None:
        components.append(FeeComponent(type="fixed_fee_schedule", schedule_id=rule.fixed_fee_schedule))
    if rule.international_surcharge_schedule is not None:
        components.append(
            FeeComponent(type="international_surcharge_schedule", schedule_id=rule.international_surcharge_schedule)
        )
    if rule.maximum_fee_schedule is not None:
        components.append(FeeComponent(type="maximum_fee_schedule", schedule_id=rule.maximum_fee_schedule))
    if (
        rule.rate_reference
        and rule.rate_reference.resolved_rate
        and rule.percentage is None
        and rule.rate_reference.resolved_rate.percentage
    ):
        components.append(FeeComponent(type="resolved_percentage", value=rule.rate_reference.resolved_rate.percentage))
    return components


def _derive_calculation_status(rule: TransactionFeeRule) -> str:
    """Return the calculability status for a rule.

    A rule is ``calculable`` when it has at least one usable fee component:
    a percentage, a resolved percentage, a fixed-fee schedule or an
    international-surcharge schedule.  Rules that are purely informational or
    have no usable component are marked ``incomplete``.
    """
    if _fee_components_for_rule(rule):
        return "calculable"
    if rule.rate_reference and rule.rate_reference.resolved_rate is None:
        return "reference_only"
    return "incomplete"


def _rule_fee_signature(rule: TransactionFeeRule) -> str:
    """Return a canonical signature of the complete fee definition carried by a rule.

    The signature includes percentage, fixed-fee schedule, international
    surcharge schedule, maximum-fee schedule, direct fixed-amount components and
    resolved reference percentage so that two rules with the same identity but
    different fee definitions are reported as conflicts.
    """
    return json.dumps(
        {
            "components": [
                _canonical_json(component.model_dump(mode="json")) for component in _fee_components_for_rule(rule)
            ],
            "maximum_fee_schedule": rule.maximum_fee_schedule,
            "calculation_status": _derive_calculation_status(rule),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _deduplicate_rules(
    rules: list[TransactionFeeRule],
    diagnostics: list[Diagnostic] | None = None,
) -> list[TransactionFeeRule]:
    """Merge equivalent rules, preserving variants and preferring resolved references.

    Rules are equivalent when their product family, variant and conditions are
    identical. Within an equivalence group we prefer:
    1. a rule with a usable rate (or a resolved reference), and
    2. a rule that carries a reference (because it ties the source and target
       together), then
    3. the first rule in source order.

    If the same selector has different fee values, the group is a genuine
    conflict: a synthetic incomplete rule is emitted and a
    ``conflicting_rule_identity`` diagnostic is added.
    """
    groups: dict[str, list[tuple[int, TransactionFeeRule]]] = {}
    for idx, rule in enumerate(rules):
        groups.setdefault(_rule_identity(rule), []).append((idx, rule))

    selected: list[TransactionFeeRule] = []
    for group in groups.values():
        if len(group) > 1:
            signatures = {_rule_fee_signature(rule) for _, rule in group}
            if len(signatures) > 1:
                # Genuine conflict: the same selector resolves to different fees.
                if diagnostics is not None:
                    first = min(group, key=lambda item: item[0])[1]
                    diagnostics.append(
                        Diagnostic(
                            type="conflicting_rule_identity",
                            rule_id=first.id,
                            label=first.label,
                            values=[_rule_fee_signature(rule) for _, rule in group],
                            sources=[rule.source for _, rule in group if rule.source],
                        )
                    )
                # Emit an incomplete placeholder for the selector so the
                # conflict is visible but no authoritative value is exposed.
                representative = min(group, key=lambda item: item[0])[1]
                selected.append(
                    representative.model_copy(
                        update={
                            "percentage": None,
                            "fixed_fee_schedule": None,
                            "international_surcharge_schedule": None,
                            "maximum_fee_schedule": None,
                            "rate_reference": None,
                            "fee_components": [],
                            "calculation_status": "incomplete",
                        }
                    )
                )
                continue

        group.sort(
            key=lambda item: (
                0 if _rule_has_rate(item[1]) else 1,
                # Prefer a source row that carries a resolved reference so the
                # reference is preserved in the final output.
                0 if _is_reference_source(item[1]) else 1,
                item[0],
            )
        )
        selected.append(group[0][1])
    # Preserve original source order when possible.
    selected.sort(key=lambda r: next((i for i, rule in enumerate(rules) if rule is r), 0))
    return selected


def _rule_sort_key(rule: TransactionFeeRule) -> tuple[int, str | None, str, str | None]:
    order = {pid: idx for idx, pid in enumerate(_PRODUCT_ORDER)}
    return (
        order.get(rule.id, 999),
        rule.variant_id or "",
        _canonical_json(rule.conditions),
        rule.label or "",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _resolve_rate_references(
    extracted_rules: list[_ExtractedRule],
    unresolved_rules: list[TransactionFeeRule | None],
    diagnostics: list[Diagnostic],
) -> None:
    """Resolve textual references against all collected rules."""
    for i, extracted in enumerate(extracted_rules):
        if not extracted.reference:
            continue
        rule = unresolved_rules[i]
        if rule is None:
            continue
        resolved, ambiguous = _resolve_reference(
            extracted.reference,
            unresolved_rules,
            source_variant_id=rule.variant_id,
            source_conditions=rule.conditions,
            source=rule.source,
        )
        if resolved:
            percentage = rule.percentage
            if resolved.percentage and percentage is None:
                percentage = resolved.percentage
            fixed_fee_schedule = rule.fixed_fee_schedule
            if resolved.fixed_fee_schedule is not None:
                fixed_fee_schedule = resolved.fixed_fee_schedule
            international_surcharge_schedule = rule.international_surcharge_schedule
            if resolved.international_surcharge_schedule is not None:
                international_surcharge_schedule = resolved.international_surcharge_schedule
            maximum_fee_schedule = rule.maximum_fee_schedule
            if resolved.maximum_fee_schedule is not None:
                maximum_fee_schedule = resolved.maximum_fee_schedule
            unresolved_rules[i] = rule.model_copy(
                update={
                    "rate_reference": RateReference(
                        reference=extracted.reference,
                        resolved_rate=resolved,
                        source=rule.source,
                    ),
                    "percentage": percentage,
                    "fixed_fee_schedule": fixed_fee_schedule,
                    "international_surcharge_schedule": international_surcharge_schedule,
                    "maximum_fee_schedule": maximum_fee_schedule,
                }
            )
        else:
            diagnostics.append(
                Diagnostic(
                    type="ambiguous_reference" if ambiguous else "unresolved_reference",
                    rule_id=rule.id,
                    label=extracted.label,
                    sources=[rule.source] if rule.source else [],
                )
            )
            if rule.percentage is None:
                unresolved_rules[i] = None


def _validate_top_level_schedule_references(
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Validate top-level schedule references and emit any remaining missing diagnostics."""
    for idx, rule in enumerate(unresolved_rules):
        updates: dict[str, Any] = {}
        if rule.fixed_fee_schedule and rule.fixed_fee_schedule not in fixed_schedules:
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="fixed_fee",
                    expected_schedule=rule.fixed_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["fixed_fee_schedule"] = None
        if (
            rule.international_surcharge_schedule
            and rule.international_surcharge_schedule not in international_schedules
        ):
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="international_surcharge",
                    expected_schedule=rule.international_surcharge_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["international_surcharge_schedule"] = None
        if rule.maximum_fee_schedule and rule.maximum_fee_schedule not in maximum_fee_schedules:
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="maximum_fee",
                    expected_schedule=rule.maximum_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["maximum_fee_schedule"] = None
        if updates:
            unresolved_rules[idx] = rule.model_copy(update=updates)


def _validate_nested_schedule_references(
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Validate nested schedule references inside resolved rates."""
    for idx, rule in enumerate(unresolved_rules):
        if not (rule.rate_reference and rule.rate_reference.resolved_rate):
            continue
        resolved = rule.rate_reference.resolved_rate
        new_rate = resolved
        if resolved.fixed_fee_schedule and resolved.fixed_fee_schedule not in fixed_schedules:
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="fixed_fee",
                    expected_schedule=resolved.fixed_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"fixed_fee_schedule": None})
        if (
            resolved.international_surcharge_schedule
            and resolved.international_surcharge_schedule not in international_schedules
        ):
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="international_surcharge",
                    expected_schedule=resolved.international_surcharge_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"international_surcharge_schedule": None})
        if resolved.maximum_fee_schedule and resolved.maximum_fee_schedule not in maximum_fee_schedules:
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="maximum_fee",
                    expected_schedule=resolved.maximum_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"maximum_fee_schedule": None})
        if new_rate is not resolved:
            unresolved_rules[idx] = rule.model_copy(
                update={"rate_reference": rule.rate_reference.model_copy(update={"resolved_rate": new_rate})}
            )


def _materialize_fee_components(
    rules: list[TransactionFeeRule],
) -> list[TransactionFeeRule]:
    """Materialize fee components and calculability status for each rule."""
    return [
        rule.model_copy(
            update={
                "calculation_status": _derive_calculation_status(rule),
                "fee_components": _fee_components_for_rule(rule),
            }
        )
        for rule in rules
    ]


def _count_inherited_schedule_references(
    rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> int:
    """Return the number of rule schedule references that point to inherited schedules."""
    inherited = 0
    for rule in rules:
        for attr, schedules in (
            ("fixed_fee_schedule", fixed_schedules),
            ("international_surcharge_schedule", international_schedules),
            ("maximum_fee_schedule", maximum_fee_schedules),
        ):
            schedule_id = getattr(rule, attr)
            if schedule_id:
                schedule = schedules.get(schedule_id)
                if schedule and schedule.origin == "inherited":
                    inherited += 1
    return inherited


def _count_inherited_schedule_objects(
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> int:
    """Return the number of schedule objects that are inherited."""
    return sum(
        1
        for schedules in (fixed_schedules, international_schedules, maximum_fee_schedules)
        for schedule in schedules.values()
        if schedule.origin == "inherited"
    )


def _count_direct_fixed_fees(rules: list[TransactionFeeRule]) -> int:
    """Return the number of direct fixed monetary fee rules."""
    fixed_schedule_count = sum(
        1
        for r in rules
        if r.percentage is None and r.fixed_fee_schedule is not None and r.international_surcharge_schedule is None
    )
    fee_component_count = sum(
        1 for r in rules for c in r.fee_components if c.type == "fixed_amount" and c.amount is not None
    )
    return fixed_schedule_count + fee_component_count


def _diagnostic_counts(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """Return counts for diagnostic types used by coverage summary."""
    types = [d.type for d in diagnostics]
    return {
        "conflicts": types.count("conflicting_schedule_entry"),
        "missing_schedules": types.count("missing_required_schedule"),
        "unresolved_references": types.count("unresolved_reference"),
        "unresolved_nested_references": types.count("unresolved_nested_reference"),
        "unknown_apm": types.count("unknown_apm_method"),
    }


def _build_coverage_summary(
    rules: list[TransactionFeeRule],
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    ignored_rows: list[UnclassifiedFeeRow],
    unclassified_rows: list[UnclassifiedFeeRow],
    ambiguous_rows: list[AmbiguousFeeRow],
    diagnostics: list[Diagnostic],
    extracted_rules: list[_ExtractedRule],
    numeric_fee_candidates: int = 0,
    unclassified_fee_candidates: int = 0,
) -> CoverageSummary:
    """Compute the classification coverage summary."""
    counts = _diagnostic_counts(diagnostics)
    reference_target_ids = {
        r.rate_reference.resolved_rate.rule_id
        for r in unresolved_rules
        if r.rate_reference and r.rate_reference.resolved_rate and r.rate_reference.resolved_rate.rule_id
    }
    extracted_apm = sum(
        len(e.conditions.get("payment_methods", []))
        for e in extracted_rules
        if e.product_id == "alternative_payment_methods"
    )

    inherited_references = _count_inherited_schedule_references(
        rules, fixed_schedules, international_schedules, maximum_fee_schedules
    )
    return CoverageSummary(
        transaction_rules=len(rules),
        calculable_rules=sum(1 for r in rules if r.calculation_status == "calculable"),
        non_calculable_rules=sum(1 for r in rules if r.calculation_status != "calculable"),
        direct_fixed_fees=_count_direct_fixed_fees(rules),
        fixed_fee_entries=sum(len(s.entries) for s in fixed_schedules.values()),
        international_surcharge_entries=sum(len(s.entries) for s in international_schedules.values()),
        maximum_fee_entries=sum(len(s.entries) for s in maximum_fee_schedules.values()),
        reference_sources=sum(1 for e in extracted_rules if e.reference),
        reference_targets=len(reference_target_ids),
        ignored=len(ignored_rows),
        unclassified=len(unclassified_rows),
        ambiguous=len(ambiguous_rows),
        conflicts=counts["conflicts"],
        missing_required_schedules=counts["missing_schedules"],
        inherited_schedules=inherited_references,
        inherited_schedule_objects=_count_inherited_schedule_objects(
            fixed_schedules, international_schedules, maximum_fee_schedules
        ),
        inherited_schedule_references=inherited_references,
        unresolved_references=counts["unresolved_references"],
        unresolved_nested_references=counts["unresolved_nested_references"],
        extracted_apm_methods=extracted_apm,
        unknown_apm_methods=counts["unknown_apm"],
        numeric_fee_candidates=numeric_fee_candidates,
        unclassified_fee_candidates=unclassified_fee_candidates,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tables(tables: list[Table], source: Source | None = None) -> DerivedFeeResult:
    """Derive product-specific transaction fee rules from normalized tables."""
    table_categories: dict[int, str | None] = {id(table): _classify_table_category(table) for table in tables}
    fixed_schedules, international_schedules, maximum_fee_schedules, schedule_diagnostics = _collect_schedules(
        tables, source=source, table_categories=table_categories
    )
    diagnostics: list[Diagnostic] = list(schedule_diagnostics)

    extracted_rules: list[_ExtractedRule] = []
    unclassified_rows: list[UnclassifiedFeeRow] = []
    ambiguous_rows: list[AmbiguousFeeRow] = []
    ignored_rows: list[UnclassifiedFeeRow] = []
    total_numeric_fee_candidates = 0
    total_unclassified_fee_candidates = 0

    for table in tables:
        category = table_categories[id(table)]
        if category in _TABLE_CATEGORY_SCHEDULE or category in {
            "commercial_rate_table",
            "online_card_rate_table",
            "goods_and_services_rate_table",
            "donation_rate_table",
            "nonprofit_rate_table",
            "apm_rate_table",
            "pos_rate_table",
            "micropayment_rate_table",
            "withdrawals_rate_table",
            "other_fees_table",
        }:
            rules, uncls, ambig, ignored, numeric_candidates, unclassified_candidates = _extract_rules_from_rate_table(
                table,
                category,
                source,
                fixed_schedules,
                international_schedules,
                maximum_fee_schedules,
            )
            extracted_rules.extend(rules)
            unclassified_rows.extend(uncls)
            ambiguous_rows.extend(ambig)
            ignored_rows.extend(ignored)
            total_numeric_fee_candidates += numeric_candidates
            total_unclassified_fee_candidates += unclassified_candidates

    # First pass: build TransactionFeeRule objects without resolving references so
    # that all candidate target rules exist for the second pass.
    unresolved_rules: list[TransactionFeeRule | None] = []
    for extracted in extracted_rules:
        prov = _provenance(
            extracted.table,
            extracted.row,
            extracted.row_index,
            source,
            original_label=extracted.label,
        )
        unresolved_rules.append(
            TransactionFeeRule(
                id=extracted.product_id,
                variant_id=extracted.variant_id,
                label=extracted.label,
                percentage=extracted.percentage,
                fixed_fee_schedule=extracted.fixed_fee_schedule,
                international_surcharge_schedule=extracted.international_surcharge_schedule,
                maximum_fee_schedule=extracted.maximum_fee_schedule,
                conditions=extracted.conditions,
                rate_reference=None,
                source=prov,
                calculation_status="calculable",
                fee_components=list(extracted.fee_components),
            )
        )
        for unknown in extracted.unknown_apm_methods or []:
            # Only emit APM diagnostics for rows that are actually classified as
            # alternative payment methods; other product rows may split into
            # unrecognised tokens but are not APM source rows.
            if extracted.product_id != "alternative_payment_methods":
                continue
            diagnostics.append(
                Diagnostic(
                    type="unknown_apm_method",
                    rule_id=extracted.product_id,
                    payment_method=unknown,
                    label=extracted.label,
                    sources=[prov],
                )
            )

    # Direct fixed-fee tables (chargebacks, refunds, withdrawals, etc.) are not
    # tied to a rate table. Create a rule per schedule for any schedule that is
    # not already referenced by a product's rate-table rule.
    referenced_schedules = {r.fixed_fee_schedule for r in unresolved_rules if r is not None and r.fixed_fee_schedule}
    direct_fixed_rules = _create_direct_fixed_fee_rules(fixed_schedules, referenced_schedules)
    unresolved_rules.extend(direct_fixed_rules)

    # Recipient service fees (e.g. UK recipient surcharge) are independent
    # surcharge schedules and not part of the commercial international surcharge.
    # Expose them as a separate rule so the fee is selectable.
    referenced_intl_schedules = {
        r.international_surcharge_schedule
        for r in unresolved_rules
        if r is not None and r.international_surcharge_schedule
    }
    for schedule_id, schedule in international_schedules.items():
        if schedule_id in referenced_intl_schedules or not schedule.entries:
            continue
        if schedule_id == "recipient_service":
            provenance = schedule.sources[0] if schedule.sources else None
            unresolved_rules.append(
                TransactionFeeRule(
                    id="recipient_service",
                    variant_id="standard",
                    label=provenance.section_heading if provenance else None,
                    percentage=None,
                    fixed_fee_schedule=None,
                    international_surcharge_schedule=schedule_id,
                    conditions={"recipient_location": "GB"},
                    rate_reference=None,
                    source=provenance,
                    calculation_status="calculable",
                    fee_components=[FeeComponent(type="international_surcharge_schedule", schedule_id=schedule_id)],
                )
            )

    # Resolve references, create explicitly inherited schedules, and validate
    # schedule references for rules that are not attached to a rate table.
    _resolve_rate_references(extracted_rules, unresolved_rules, diagnostics)
    resolved_rules: list[TransactionFeeRule] = [r for r in unresolved_rules if r is not None]
    _resolve_schedule_inheritance(
        extracted_rules,
        resolved_rules,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        diagnostics,
    )
    _validate_inheritance_priorities(
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        diagnostics,
    )
    _validate_top_level_schedule_references(
        resolved_rules, fixed_schedules, international_schedules, maximum_fee_schedules, diagnostics
    )
    _validate_nested_schedule_references(
        resolved_rules, fixed_schedules, international_schedules, maximum_fee_schedules, diagnostics
    )

    # Merge equivalent rules and preserve legitimate variants.
    transaction_rules = _deduplicate_rules(resolved_rules, diagnostics)
    transaction_rules.sort(key=_rule_sort_key)

    # Materialize fee components and calculability status for each rule.
    transaction_rules = _materialize_fee_components(transaction_rules)

    # Currency conversion.
    currency_conversion = None
    for table in tables:
        if table_categories[id(table)] == "currency_conversion_table":
            for row in table.rows:
                pct = _first_percentage(row)
                if pct:
                    currency_conversion = CurrencyConversion(spread_percentage=pct)
                    break
            if currency_conversion:
                break

    coverage = _build_coverage_summary(
        transaction_rules,
        resolved_rules,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        ignored_rows,
        unclassified_rows,
        ambiguous_rows,
        diagnostics,
        extracted_rules,
        numeric_fee_candidates=total_numeric_fee_candidates,
        unclassified_fee_candidates=total_unclassified_fee_candidates,
    )

    status = _derive_status(
        transaction_rules,
        unclassified_rows,
        ambiguous_rows,
        ignored_rows,
        diagnostics,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
    )

    return DerivedFeeResult(
        status=status,
        transaction_fee_rules=transaction_rules,
        fixed_fee_schedules=fixed_schedules,
        international_surcharge_schedules=international_schedules,
        maximum_fee_schedules=maximum_fee_schedules,
        currency_conversion=currency_conversion,
        unclassified_fee_rows=unclassified_rows,
        ambiguous_rows=ambiguous_rows,
        ignored_rows=ignored_rows,
        diagnostics=diagnostics,
        coverage_summary=coverage,
    )
