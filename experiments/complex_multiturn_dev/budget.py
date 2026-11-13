"""Fail-closed, append-only budget for serial DeepSeek Flash calls.

Rates are peak mainland-CNY prices checked on 2026-11-13. Verified cache usage
uses the peak cache rate; missing details stay at the all-miss upper bound.
Unknown outcomes retain their full reservation.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4


class BudgetUnavailable(RuntimeError):
    pass


class BudgetLedger:
    INPUT_PER_MILLION = Decimal("2")
    CACHE_HIT_PER_MILLION = Decimal("0.04")
    OUTPUT_PER_MILLION = Decimal("8")
    ENVELOPE_TOKENS = 4096

    def __init__(self, path: Path, *, limit_cny: Decimal):
        self.path = Path(path)
        self.limit = Decimal(limit_cny)
        if not self.limit.is_finite() or self.limit <= 0 or self.limit > Decimal("40"):
            raise ValueError("budget must be positive and no more than 40 CNY")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self):
        open_reservations = {}
        spent = Decimal(0)
        if not self.path.exists():
            return spent, open_reservations
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                amount = Decimal(record["amount_cny"])
                if not amount.is_finite() or amount < 0:
                    raise ValueError("invalid budget amount")
                ident = record["id"]
                if record["kind"] == "reserve" and ident not in open_reservations:
                    open_reservations[ident] = amount
                elif record["kind"] == "settle" and ident in open_reservations:
                    if amount > open_reservations.pop(ident):
                        raise ValueError("settlement exceeds reservation")
                    spent += self._settled_upper_bound(record, amount)
                else:
                    raise ValueError("invalid ledger sequence")
        except (KeyError, TypeError, InvalidOperation, json.JSONDecodeError) as exc:
            raise BudgetUnavailable("budget ledger is invalid") from exc
        return spent, open_reservations

    @classmethod
    def _settled_upper_bound(cls, record, recorded_amount):
        """Use provider-reported hits only when all counts and the old debit agree."""
        if (record.get("accounting") != "peak_all_cache_misses_upper_bound"
                or record.get("cache_usage_status") != "reported"):
            return recorded_amount
        values = [record.get(key) for key in
                  ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens")]
        if any(type(value) is not int or value < 0 for value in values):
            return recorded_amount
        prompt, completion, hit, miss = values
        if hit + miss != prompt:
            return recorded_amount
        all_miss = (Decimal(prompt) * cls.INPUT_PER_MILLION +
                    Decimal(completion) * cls.OUTPUT_PER_MILLION) / Decimal(1000000)
        if all_miss != recorded_amount:
            return recorded_amount
        return (Decimal(hit) * cls.CACHE_HIT_PER_MILLION + Decimal(miss) * cls.INPUT_PER_MILLION +
                Decimal(completion) * cls.OUTPUT_PER_MILLION) / Decimal(1000000)

    def _append(self, record):
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def spent_or_reserved(self):
        spent, pending = self._read()
        return spent + sum(pending.values(), Decimal(0))

    def reserve(self, *, request_bytes: int, max_output_tokens: int):
        if type(request_bytes) is not int or request_bytes < 0:
            raise ValueError("request byte count is required")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise BudgetUnavailable("finite maximum output is required")
        spent, pending = self._read()
        if pending:
            raise BudgetUnavailable("unsettled previous request; reconcile before continuing")
        # UTF-8 byte count bounds raw content tokens; add overhead for API framing.
        input_bound = request_bytes + self.ENVELOPE_TOKENS
        amount = (Decimal(input_bound) * self.INPUT_PER_MILLION +
                  Decimal(max_output_tokens) * self.OUTPUT_PER_MILLION) / Decimal(1000000)
        if spent + amount > self.limit:
            raise BudgetUnavailable("next request could exceed the budget limit")
        ident = uuid4().hex
        self._append({"kind": "reserve", "id": ident, "amount_cny": str(amount),
                      "input_token_upper_bound": input_bound,
                      "max_output_tokens": max_output_tokens,
                      "pricing": "deepseek_flash_cny_peak_20261113"})
        return ident

    def reserve_fixed(self, *, amount_cny: Decimal, purpose: str):
        amount = Decimal(amount_cny)
        prices = {"search": Decimal("0.005"), "fetch": Decimal("0.005"),
                  "amap_search": Decimal("0.003")}
        if purpose not in prices or amount != prices[purpose]:
            raise BudgetUnavailable("unknown fixed-price request")
        spent, pending = self._read()
        if pending:
            raise BudgetUnavailable("unsettled previous request; reconcile before continuing")
        if spent + amount > self.limit:
            raise BudgetUnavailable("next request could exceed the budget limit")
        ident = uuid4().hex
        self._append({"kind": "reserve", "id": ident, "amount_cny": str(amount),
                      "purpose": purpose, "pricing": "provider_fixed_cny_20261113"})
        return ident

    def settle_fixed(self, ident: str):
        _, pending = self._read()
        if ident not in pending:
            raise BudgetUnavailable("reservation is unavailable")
        # Charge the maximum even if the request failed; this is an upper bound.
        self._append({"kind": "settle", "id": ident, "amount_cny": str(pending[ident]),
                      "accounting": "conservative_fixed_upper_bound"})

    def settle_uncertain_at_reserved_cap(self, ident: str, *, reason: str, approved: bool):
        """Book a single uncertain request at its full cap after explicit user approval."""
        if approved is not True:
            raise BudgetUnavailable("explicit user approval is required")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise BudgetUnavailable("a concise reconciliation reason is required")
        _, pending = self._read()
        if len(pending) != 1 or ident not in pending:
            raise BudgetUnavailable("the single reservation must be identified")
        self._append({"kind": "settle", "id": ident, "amount_cny": str(pending[ident]),
                      "accounting": "approved_uncertain_usage_reserved_upper_bound",
                      "usage": "unknown", "reason": reason.strip(),
                      "reconciled_at_utc": datetime.now(timezone.utc).isoformat()})

    def settle(self, ident: str, *, prompt_tokens: int, completion_tokens: int,
               cache_hit_tokens=None, cache_miss_tokens=None):
        if (type(prompt_tokens) is not int or prompt_tokens < 0 or
                type(completion_tokens) is not int or completion_tokens < 0):
            raise BudgetUnavailable("provider usage is unavailable")
        _, pending = self._read()
        if ident not in pending:
            raise BudgetUnavailable("reservation is unavailable")
        amount = (Decimal(prompt_tokens) * self.INPUT_PER_MILLION +
                  Decimal(completion_tokens) * self.OUTPUT_PER_MILLION) / Decimal(1000000)
        if amount > pending[ident]:
            raise BudgetUnavailable("actual usage exceeds the reserved amount")
        record = {"kind": "settle", "id": ident, "amount_cny": str(amount),
                  "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "usage_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "accounting": "peak_all_cache_misses_upper_bound"}
        if (type(cache_hit_tokens) is int and type(cache_miss_tokens) is int
                and cache_hit_tokens >= 0 and cache_miss_tokens >= 0
                and cache_hit_tokens + cache_miss_tokens == prompt_tokens):
            record.update(cache_hit_tokens=cache_hit_tokens, cache_miss_tokens=cache_miss_tokens,
                          cache_usage_status="reported")
        else:
            record["cache_usage_status"] = "unavailable_or_inconsistent"
        self._append(record)
