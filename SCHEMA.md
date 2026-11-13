# NimbusPay Triage Labelling Spec (v2, current)

NimbusPay is a fictional payments app. Every incoming support ticket is turned into one JSON object, the *triage record*. This document is the single source of truth for what a correct record looks like. **All dev and test labels follow this spec exactly.**

## Input

```
Received: 2026-08-14 (Fri)
Source: email

<the customer's message>
```

`Received` is the date the ticket arrived. `Source` is one of `app-chat`, `email`, `twitter`, `ivr-transcript`.

## Output

A single JSON object with **exactly these 8 keys**. Use `null` when a value is not present in the ticket.

```json
{"category": "refund", "priority": "P2", "amount_paise": 2936300, "txn_date": "2026-09-13", "txn_id": "NP5972758064", "channel": "netbanking", "language": "en", "needs_human": false}
```

| Key | Type | Allowed values |
|---|---|---|
| `category` | string | `payment_failed`, `refund`, `fraud`, `account_access`, `kyc`, `offers`, `other` |
| `priority` | string | `P1`, `P2`, `P3` |
| `amount_paise` | integer or null | amount in paise (₹1 = 100 paise) |
| `txn_date` | string or null | `YYYY-MM-DD` |
| `txn_id` | string or null | `NP` followed by 10 digits |
| `channel` | string or null | `upi`, `card`, `netbanking`, `wallet` |
| `language` | string | `en`, `hi-en` |
| `needs_human` | boolean | `true`, `false` |

## Rules

### category
| Value | Use when |
|---|---|
| `fraud` | The customer says a transaction (payment, cash withdrawal, direct debit) happened that **they did not make, approve or recognise**. This wins over every other category. |
| `payment_failed` | A payment, transfer or top-up the customer made failed, was declined, is stuck, or did not reach the recipient. |
| `refund` | Money is owed back: asking for a refund, a cancelled order, a return, the same transaction charged more than once, a refund that has not arrived. |
| `account_access` | Cannot get in: locked or blocked account, blocked PIN, forgotten passcode or password, OTP not received, reset problems. |
| `kyc` | Identity verification: how to verify, why it is needed, verification that failed or is pending, documents, Aadhaar/PAN, video KYC. |
| `offers` | Cashback, promo codes, coupons, scratch cards, referral bonuses, reward points. |
| `other` | Anything else: card delivery, exchange rates, closing the account, general how-to questions. |

### priority (apply in this order)
1. **`P1`** if any of these is true:
   - `category` is `fraud`
   - the disputed amount is **₹50,000 or more**
   - the customer makes a **legal threat**: mentions RBI, an ombudsman, consumer court, a lawyer, a legal notice or legal action
2. **`P2`** if not P1 and either:
   - `category` is `payment_failed` or `refund` **and** the amount is **₹5,000 or more**
   - `category` is `account_access`
3. **`P3`** otherwise.

### Transaction fields
`amount_paise`, `txn_date`, `txn_id` and `channel` describe one specific transaction. For `account_access`, `kyc` and `other` tickets they are always `null`.

### amount_paise
- The amount of the transaction or claim the ticket is **about**, converted to paise as an integer.
- Ignore other amounts: account balance, limits, fees.
- `k` means thousand (`1.3k` = ₹1,300). `lakh` means 100,000 (`1.5 lakh` = ₹1,50,000).
- `null` if no such amount is stated.

### txn_date
- The date of the transaction the ticket is about, as `YYYY-MM-DD`.
- Resolve relative dates ("yesterday", "3 days ago", "kal", "parso") against the `Received` date.
- If the year is missing, use the most recent such date on or before the `Received` date.
- Dates inside quoted earlier emails (lines starting with `>`) are not the transaction date.
- `null` if the customer does not say when it happened.

### txn_id
- NimbusPay transaction ids are `NP` + 10 digits. Normalize to uppercase with no spaces, hyphens or symbols: `np 12345 67890` → `NP1234567890`.
- Other reference numbers (ticket numbers, bank reference numbers, order numbers) are **not** transaction ids.
- `null` if no NimbusPay transaction id is given.

### channel
- The payment mode of the transaction, **only when it is explicitly stated**: `upi`, `card` (credit card, debit card, "card payment", Visa, Mastercard), `netbanking` (also "internet banking"), `wallet`.
- `null` otherwise.

### language
- `hi-en` if the customer's message is written in Hinglish (Hindi in Latin script, possibly mixed with English words).
- `en` otherwise.

### needs_human
`true` if `priority` is `P1`, **or** the customer says they have already contacted support about this issue before (for example "third time", "again and again", "baar baar"). Otherwise `false`.
