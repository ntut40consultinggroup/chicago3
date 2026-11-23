"""ETF 持股來源。目前支援：統一投信（ezmoney，僅最新一天）、復華投信（fhtrust，可指定日期）。"""
from . import ezmoney, fhtrust
from .base import COLUMNS, Holdings

SOURCES = {"ezmoney": ezmoney, "fhtrust": fhtrust}
