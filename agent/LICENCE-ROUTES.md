# Licence routes for a bot-usable `mkt+r24` (2026-11-18)

Question: which market-data source may train a 24 h cross-sectional ranker whose output sizes the owner's own trades, and does the
`mkt+r24` trade-day tilt survive on it. Terms were read from the live pages (or a 2026 Wayback copy when the live page refused a
script; marked) on 2026-11-18. Quotes are verbatim. This records what the text says; it is not legal advice and draws no legal
conclusion beyond the text. Raw copies were kept in the session scratchpad, not committed.

## 1. Survey

| source | hourly bars, cross-section, history | trading use (text) | model training (text) | sharing a derived model (text) | cost |
|---|---|---|---|---|---|
| Binance Vision (current research tier) | 1 h klines, about 690 USDT pairs, from 2017 | prohibited for "live proprietary trading execution" (4.2) | models are governed derivatives (2.4), non-commercial only | CC BY-NC-SA 4.0 only (4.5) | free |
| Alpaca Market Data, crypto | 1 h bars from Alpaca's crypto venue; 37 of 58 probed USD pairs have bars; hourly from 2021-01 for 12 (BTC, ETH, SOL, DOGE, LTC, BCH, LINK, UNI, ...), about 20 by 2022, many only from 2026-02 (measured 2026-11-18, market-data GETs only) | "Content is provided exclusively for personal and noncommercial access and use" | not addressed | no copying / distribution "for any commercial enterprise" without consent | free (Basic) |
| Coinbase Exchange market data | 1 h candles, public REST; about 250-300 USD pairs now, roughly 50-60 by 2020 (not fetched) | licence is "exclusively for you or your entity's personal or research purposes" | expressly barred: "develop, train, fine-tune, ... any artificial intelligence or machine learning model, algorithm ... or other automated system" (3.5) | no "Derived Works" to any third party (3.2) | free |
| Kraken (OHLCVT archive, public API) | 1 h OHLCVT per pair from its first trade to 2026-09-30, free download; 741 USD pairs in the archive; after the prereg liquidity filter 156 listed by 2022-12, of which 15 by 2019-01, 27 by 2020-01, 48 by 2021-01, 98 by 2022-01 (measured) | content usable "only for your own benefit"; no clause on trading with it | not addressed | no commercial exploitation or making it "available to any third party" | free |
| Bitstamp | 1 h OHLC via public API from listing; about 80-100 USD pairs now, about 15 in 2020 (not fetched) | site data for "personal, informational and/or non-commercial use" | not addressed; no "scraped, collected ... whether manual or automated" | barred | free |
| Gemini | 1 h candles, recent window only; about 70 USD pairs now (not fetched) | "personal and/or internal use" and "analysis of prices and markets"; "commercial purpose" -> bizdev | not addressed | redistribution barred | free; commercial by agreement |
| CryptoCompare / CoinDesk Data | aggregated 1 h OHLCV, thousands of pairs, long history | "own internal use only"; "must not be used in connection with any commercial product or service"; data "not intended for trading purposes" | not addressed | barred | free tier retired 2026-05-21; paid by quote |
| CoinGecko | aggregated hourly price / volume / cap (no hourly OHLC); Demo 1 year, Basic 2 years, Analyst+ from 2018; thousands of coins | paid plans list "License: Commercial"; reliance for "trading" at own risk | not addressed | may "charge for your services and products that incorporate" the API; no re-distributing API access | Demo free (attribution); Basic USD 35/mo; Analyst USD 129/mo |
| Tardis.dev | tick trades / books for Binance spot, Kraken, OKX, Coinbase etc.; candles derivable | "Permitted Use ... includes trading, investment, research and risk management" | allowed: "may also develop, train and fine-tune its own models on the Data ... provided that only Customer Users can use them"; Coinbase data excluded (cl. 23) | no model or weights to anyone outside Customer Users (9.5(b)) | spot Solo USD 900/mo, Academic 450, Professional 1,350; all exchanges 1,200 Solo |
| Kaiko | aggregated and per-venue OHLCV, broad history | "Client's internal business purposes, including the creation of derived works to be used internally" | not addressed beyond internal derived works | client-facing derived products need consent | enterprise, by quote |
| KuCoin (public REST klines; the owner's trading venue; wave 4e) | 1 h klines per USDT pair from its listing to now, 1,500 rows per request; 808 USDT pairs listed now, 232 of them listed by 2022-12 and all pass the liquidity filter (31 by 2020-01, 67 by 2021-01, 187 by 2022-01); delisted pairs are not served (measured) | no clause on trading with the data; Terms of Use bar "commercial use of the Platform or any content thereof" and any use "not explicitly permitted by the Terms of Use" (art. 91); the API page offers the API for "quantitative trading systems" | not addressed | no "derivatives of the Platform or content thereof"; no systematic "collections, compilations, databases" without written permission (art. 91) | free |
| OKX (public REST candles; wave 4e) | 1 h candles, 100 rows per request, hourly history from 2018-01 for old pairs; 491 USDT pairs listed now (pair list measured; bars fetched for 3 coins only) | Market Data "solely for your own personal, non-commercial trading and account management purposes" (9.4); trading "for your own account, even if profitable, does not constitute commercial resale" (9.3) | barred for a model "that is intended to be commercialized or made available to third parties" (9.4(c)) | no making Market Data "available to any third party"; no "competing data product" (9.4) | free; "separate data licensing agreement" for large-scale or commercial access |

### Quotes and URLs

Alpaca. `https://files.alpaca.markets/disclosures/library/TermsAndConditions.pdf` (linked from `alpaca.markets/disclosures`):
- "Content includes ... (3) market data such as quotations ..."
- "Content is provided exclusively for personal and noncommercial access and use. No part of the Service or Content may be copied,
  reproduced, republished, uploaded, posted, publicly displayed, encoded, translated, transmitted or distributed in any way
  (including "mirroring") to any other computer, server, web site or other medium for publication or distribution or for any
  commercial enterprise, without Alpaca's express prior written consent."
- Crypto data docs (`docs.alpaca.markets/docs/historical-crypto-data-1`) say the crypto endpoints now serve Alpaca's own venue.

Coinbase. `https://www.coinbase.com/legal/market_data`, "Last updated: August 7, 2026" (live page returns 403 to scripts; read from
the Wayback copy of 2026-09-02). The Exchange docs bind Market Data API users to these terms.
- 2: "Your use of Market Data is exclusively for you or your entity's personal or research purposes and may not be used to build an
  application intended for use by end users other than for you or your officers/employees."
- 3.2: no redistribution of "the Market Data--or any data, charts, analytics, research, or other works based on, referring to, or
  derived from the Market Data ("Derived Works") -- to any third party outside of your organization".
- 3.5: "Use any Market Data to, or permit or authorize any third party to, in whole or in part, develop, train, fine-tune, teach,
  validate, benchmark, or otherwise improve any artificial intelligence or machine learning model, algorithm, chatbot, agent, or
  other automated system, whether generative or non-generative, or for any other purpose related to the research, development, or
  operation of artificial intelligence technologies."

Kraken. `https://www.kraken.com/legal/global-terms`, "Last updated: November 6, 2026"; no separate API or market-data terms are listed
on `kraken.com/legal`. Archive: `https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data`
("Last updated: November 2, 2026"), which states no licence or use terms of its own.
- 8: "We or our licensors own (1) our services and Platforms, (2) all content, materials, software, and trademarks found on them ...
  (collectively, "Our Content"). So long as you comply with these Terms, you are permitted to use our services, and Our Content made
  available to you as part of our services, but only for your own benefit. We can take away this permission at any time for any
  reason. ... If you wish to use Our Content for any other purpose you must seek prior permission".
- 9: you will not "use (except as expressly permitted in these Terms), license, sublicense, sell, resell, transfer, assign,
  distribute or otherwise commercially exploit or make available to any third party Our Content in any way", "use any web scraping,
  web harvesting, or data extraction methods to extract any data from Our Content", "create, use, operate, or employ any bots,
  robots, parsers, spiders, scripts, programs, routines, or any other forms of automation to engage in any activity on Our Content",
  "access or use Our Content to build or support products or services competitive to our products or services".
- The trading rules (`kraken.com/legal/exchangetradingrules`) describe API access to the platform; they say nothing on data use.
- Reading: no machine-learning clause and no trading-use clause; use is limited to "your own benefit"; the anti-automation bullet is
  worded broadly enough to read on scripted use of the site, while Kraken itself publishes the archive and the APIs.

Bitstamp. `https://www.bitstamp.net/terms-of-use/` (bot-protected; Wayback copy of 2026-10-27 of `/legal/terms-of-use/ltd/`, "last
updated on May 30, 2025"):
- "You may view, print and/or download a copy of the Materials from this Site on any single computer solely for your personal,
  informational and/or non-commercial use".
- "The software, text, images, graphics, data, prices, trades, charts, graphs, video and audio used on this Site belong to Bitstamp.
  The Trademarks and Material should not be copied, reproduced, modified, republished, uploaded, posted, transmitted, scraped,
  collected or distributed in any form or by any means, whether manual or automated."

Gemini. API Agreement `https://www.gemini.com/legal/api-agreement` and Market Data Agreement
`https://www.gemini.com/legal/market-data-agreement`, both dated October 12, 2025:
- "Permitted Use means your use of Gemini Market Data solely for: (i) personal and/or internal use; (ii) general informational
  purposes; or (iii) analysis of prices and markets and financial products relating thereto".
- Prohibited Use includes using it to "develop, create, register, issue, price, list ... any new or existing investment product,
  financial product ..." and to "distribute, redistribute, disseminate ... sell, resell, license, or sublicense Gemini Market Data".
- "If you would like to use Gemini Market Data for a commercial purpose, please email bizdev@gemini.com."

CryptoCompare / CoinDesk Data. Site terms `https://www.cryptocompare.com/terms-conditions/` 5.1: "The Site and the Content is for
Your personal and non-commercial use only." API licence `https://data.coindesk.com/api-licence-agreement`: data "for your own
internal use (meaning you must not share it with anyone else)"; "An important restriction to note, even when using the data for your
own internal use, is that it must not be used in connection with any commercial product or service"; 10.1.1 "the API Data is not
intended for trading purposes". Free tier: `https://data.coindesk.com/blogs/changes-to-coindesk-data-indices-api-free-tier-access`
"As of May 21, 2026, we will be retiring our free API tier".

CoinGecko. API terms `https://www.coingecko.com/en/api_terms` (Wayback copy of 2026-09-27; live page 403): 4.1.1 the scope of the
licence "will be dependent on the usage plan that you select"; 4.1.6 "You are entitled to charge for your services and products that
incorporate or integrates our CoinGecko API. However, you are not permitted to sell, rent, lease, sub-license, re-distribute or
syndicate access to the CoinGecko API"; 4.3 attribution "Powered by CoinGecko". Pricing `https://www.coingecko.com/en/api/pricing`
(Wayback 2026-11-17): paid plans "License: Commercial"; hourly history "1 year" (Demo) / "2 years" (Basic) / "from 2018"
(Analyst and up). No machine-learning clause in the terms text.

Tardis.dev. `https://docs.tardis.dev/legal/terms-of-service`, "Last modification date: 2026-10-28":
- Permitted Use: "use by Customer Users for the Customer's internal business, research, educational or personal purposes. This
  includes trading, investment, research and risk management for the Customer's clients."
- 9.5: "The Customer may also develop, train and fine-tune its own models on the Data and Manipulated Data, provided that only
  Customer Users can use them." It shall not "(b) sell, license, publish or otherwise make available to anyone other than Customer
  Users any artificial intelligence or machine learning model developed, trained or distilled, directly or indirectly, using the
  Data". "Clause 23 bars any artificial intelligence or machine learning use of Coinbase Data, and other applicable licensor
  restrictions also apply."
- 9.4: the licence "is perpetual for Data downloaded in accordance with this Agreement during paid or free access".
- Prices (`https://tardis.dev/`): spot USD 450 / 900 / 1,350 / 3,500 per month (Academic / Solo / Professional / Business), all
  exchanges 650 / 1,200 / 2,200 / 6,000; 4 years of history on yearly billing below Business.

Kaiko. General terms `https://www.kaiko.com/terms`, 1.1(a)(i): "use solely in connection with Client's internal business purposes,
including the creation of derived works to be used internally by Client solely for its general business purposes"; 1.1(b): "To the
extent, Client seeks to commercialize any derived works into a client-facing product, Client shall contact Supplier". Pricing by
quote (`https://www.kaiko.com/about-kaiko/pricing-and-contracts`).

KuCoin (added for wave 4e; typographic quotes normalised here and for OKX). Terms of Use `https://www.kucoin.com/legal/terms-of-use`,
"Last updated: 10/29/2026 11:30:00" (live page). The public site links no separate API or market-data terms, and the API docs index
(`https://www.kucoin.com/docs-new/llms.txt`) lists none; whatever the API-key creation page shows behind login was not read. The
Terms of Use have no clause on APIs, machine learning or model training. They define the Platform as KuCoin (www.kucoin.com).
- Art. 89: "Intellectual property rights, including trademark rights, patent rights, copyrights, trade secrets, and so on, to all the
  content on the Platform, including works, pictures, archives, information, materials, platform architecture, the arrangement of the
  platform screen, platform design, text and graphics, software compilation, the relevant source code and software, are legally owned
  by the Platform or other rights-holders."
- Art. 90: "Without the written consent of the Platform or other rights-holders, no one is allowed to use, modify, copy, publicly
  disseminate, change, distribute, release or publicly publish the Platform program or content without due authorization."
- Art. 91: "The Users are not allowed to download (except for web page caches) or modify the Platform or any part thereof without the
  explicit written consent of the Platform. The Users are not allowed to resell or make commercial use of the Platform or any content
  thereof; the Users may not: collect and use product catalogues, descriptions and prices, make any derivatives of the Platform or
  content thereof; download or copy account information or use any data collection robots or similar data collection and extraction
  tools for other commercial interests. Without the written permission of the Platform, it is strictly prohibited to systematically
  obtain the content of the Platform to directly or indirectly create or edit collections, compilations, databases, or personal name
  and address records (whether or not through robots, spiders, automatic instruments, or manual operations). In addition, it is
  strictly prohibited to use the content and materials on the Platform for any purpose that is not explicitly permitted by the Terms
  of Use."
- Art. 92: "Without the explicit written consent of the Platform, the Platform website or any part thereof may not be copied,
  photocopied, duplicated, sold, resold, accessed or otherwise used for any commercial purpose."
- API product page `https://www.kucoin.com/api` (marketing text, not terms): "High-performance, low-latency, full-featured APIs for
  fast, secure integration with the KuCoin ecosystem. Designed to support stable, reliable quantitative trading systems."; its FAQ:
  "You can download KuCoin's historical market data from our Historical Data page for backtesting and research."
- Klines endpoint docs (`https://www.kucoin.com/docs-new/rest/spot-trading/market-data/get-klines.md`): rate-limit pool "Public",
  weight 3; "Klines data may be incomplete. No data is published for intervals where there are no ticks."
- Reading: no machine-learning clause, no trading-use clause, no API-specific grant. Articles 89-92 are written for the Platform and
  its content: no commercial use of the content, no collecting of prices, no extraction tools "for other commercial interests", no
  systematic building of databases without written permission, no use "for any purpose that is not explicitly permitted". KuCoin
  itself publishes the API, its rate limits and a history download "for backtesting and research". The text does not say whether API
  bars pulled to size the owner's own orders on KuCoin fall inside or outside these articles.

OKX (added for wave 4e). API Agreement `https://www.okx.com/help/okx-api-agreement`, "Published on Mar 26, 2026", "Last Updated: 28
July 2026" (live page, served here as the `en-us` copy; the `en-au`, `en-gb`, `en-eu` and `en-sg` copies carry the same date and the
sentences quoted from 9.3 and 9.4). Terms of Service `https://www.okx.com/help/terms-of-service`, "Last Updated: 29 October 2026",
4.3: "For the API Services, please refer to the OKX API Agreement".
- 1.8: "Market Data" means "any price, order book, trade history, ticker, funding rate, index, volatility surface, open interest,
  trading volume, or other market information made available through the API Services or through OKX's publicly accessible
  endpoints, whether accessed with or without authentication."
- 3.3: "Users may access the API Services for lawful trading, account management, market data consumption, and integration purposes."
- 9.2: a licence "to access and use the API Services solely: (a) for your own internal purposes; (b) in connection with managing and
  operating your own OKX account; and (c) in a manner consistent with this Agreement and the OKX ToS."
- 9.3: "For the avoidance of doubt, use of the API Services to execute trades for your own account, even if profitable, does not
  constitute commercial resale."
- 9.4: "Market Data obtained through the API Services is proprietary to OKX or its data licensors. You may access and use Market Data
  solely for your own personal, non-commercial trading and account management purposes. You may not: (a) resell, redistribute,
  publish, display, or otherwise make Market Data available to any third party, whether for commercial gain or otherwise, without
  OKX's prior written consent; (b) use Market Data to build, operate, or contribute to any competing data product, market data
  service, financial data aggregator, price feed, or analytics platform; (c) use Market Data to train, fine-tune, or evaluate any
  machine learning model or AI system that is intended to be commercialized or made available to third parties; or (d) use Market
  Data in any manner that competes with or substitutes for OKX's commercial data licensing offerings."
- 9.4, public endpoints: "The restrictions in this Section 9.4 apply equally to Market Data accessed through public endpoints as to
  Market Data accessed through authenticated endpoints."
- 9.4, automated collection: "You must not use automated scripts, bots, web scrapers, or any other automated means to extract Market
  Data from OKX's public endpoints or website at a rate or scale that: (a) exceeds what is reasonably necessary for your personal
  trading use; (b) is intended to circumvent OKX's commercial data licensing requirements; or (c) imposes unreasonable load on OKX's
  infrastructure. Users who require high-frequency or large-scale access to Market Data for commercial or institutional purposes must
  obtain a separate data licensing agreement with OKX."
- 13.3: "Users must not use data obtained through the API Services to train or fine-tune any AI model in a manner that incorporates or
  exposes other users' personal data or OKX's proprietary Market Data contrary to Section 9.4."
- ToS 8.1: "We grant you a limited, non-exclusive, non-transferable license, subject to the Terms, to access and use our OKX Platform
  and Services, solely for purposes permitted by us. You agree that you will not copy, transmit, distribute, sell, license, reverse
  engineer, modify, publish, or participate in the transfer or sale of, create derivative works from, or in any other way, exploit
  any of our products and Services."
- Reading: the only text in this survey that speaks to model training on a venue's own API data, and it does so conditionally: the
  bar in 9.4(c) is on a model "intended to be commercialized or made available to third parties". Market Data is for "your own
  personal, non-commercial trading and account management purposes"; 9.3 says trading one's own account "even if profitable" is not
  commercial resale (that sentence is about the API Services). Automated extraction beyond what is "reasonably necessary for your
  personal trading use" needs a data licence.

## 2. Which sources reach the needed cross-section

`mkt+r24` needs hourly OHLC per symbol, a market pool large enough for the mean / dispersion / breadth series, groups of at least 8
(32 sampled) symbols alive at the same hour for the correlation loss, and 6,200 h of history before each origin. On Binance the
training pool was 333 symbols. No free source with usable terms reaches 100+ symbols hourly from 2020:
- Kraken: 27 symbols by 2020-01, 98 by 2022-01, 156 by 2022-12 (after the liquidity filter). The largest free cross-section here.
- Alpaca: about 20 usable USD pairs before 2023. Too small.
- Coinbase is large enough (50-60 USD pairs in 2020, a few hundred later) but 3.5 bars ML training outright.
- Gemini: no long hourly history; Bitstamp: about 15 USD pairs in 2020 and its terms bar automated collection.
- 100+ from 2020 needs a paid source: Tardis (Binance spot trades from 2019, own-model training and trading allowed, internal use
  only), CoinGecko Analyst (thousands of coins hourly from 2018, commercial licence, but close / volume only, no hourly OHLC), or Kaiko.

## 3. Test: `mkt+r24` retrained on Kraken (`agent/PREREG-LIC.md`)

Kraken OHLCVT full archive (5 parts, SHA-256 checked against Kraken's published sums), 60-minute files, USD pairs, canonical names
`<BASE>USDT`, cache in `runs-agent/lic/cache` (never published). Universe by the protocol's rule: pool 156, evaluation 47, held-out
12, train 144. Same recipe as wave 4 (`train/rank24.py`, folds F1-F4, 1,500 steps, final EMA, lambda 0.1, seeds 0 and 1; 2 x 326 s
GPU). Forecasts from Kraken bars only; Kraken has 21 of the 30 R9-R24 coins (missing HBAR, IOTA, NEO, ONE, RVN, TFUEL, VET, ZEN,
ZIL: not listed by 2022 or never) and 26 of the 36 ALGOS coins; bars end 2026-09-30, so October 2026 has no tilt.

Forecast quality:

| measure | Kraken-trained `mkt+r24` | Binance-trained `mkt+r24` (same cells) |
|---|---|---|
| validation 24 h rank IC, Kraken seen symbols (28), 8,515 origins | +0.063 [+0.045, +0.081] | n/a (registered Binance `ohlc+cal` on the same rows: +0.033, d +0.030 [+0.008, +0.053]) |
| next-day IC on the ALGOS harness (14 universes, 19,484 universe-days) | +0.060 [+0.046, +0.076] | +0.060 [+0.045, +0.074] |
| rho with the trend score | +0.004 [-0.024, +0.028] | -0.129 [-0.151, -0.106] |
| per-day rank correlation of the two forecasts (shared cells, 1,369 days) | 0.67 (R9-R24) / 0.69 (ALGOS) | |

Tilt on R9-R24 (2023-01 .. 2026-10, 20 placebos each, base arms reused from `runs-agent/r24/confirm`):

| forecast / form | cost | mean d Sharpe [95 %] | share | universes + | placebo p95 / max | turnover | ruin dd60 (M11) | class |
|---|---|---|---|---|---|---|---|---|
| Kraken `gt` (co-primary) | c7 | +0.065 [-0.049, +0.180] | 68 % | 12/16 | -0.016 / -0.006 | -3.0 % | 41.1 -> 28.6 | NEUTRAL |
| Kraken `gt` | c10 | +0.062 [-0.051, +0.177] | 67 % | 12/16 | -0.018 / -0.009 | -3.0 % | 41.2 -> 28.8 | NEUTRAL |
| Kraken `gtr` risk-matched (co-primary) | c7 | +0.031 [-0.053, +0.120] | 60 % | 11/16 | -0.023 / -0.021 | +1.9 % | 41.1 -> 35.6 | NEUTRAL |
| Kraken `gtr` | c10 | +0.028 [-0.056, +0.118] | 59 % | 11/16 | -0.026 / -0.024 | +1.9 % | 41.2 -> 35.9 | NEUTRAL |
| reference: Binance `mkt+r24`, Kraken's cells only, `gt` | c7 | +0.038 [-0.076, +0.156] | 62 % | 11/16 | -0.014 / -0.013 | -4.7 % | 41.1 -> 25.6 | NEUTRAL |
| reference | c10 | +0.035 [-0.079, +0.154] | 61 % | 11/16 | -0.017 / -0.016 | -4.7 % | 41.2 -> 25.7 | NEUTRAL |
| (wave 4c, full coverage) Binance `gt` | c7 | +0.141 [+0.024, +0.263] | 81 % | 15/16 | +0.028 / +0.029 | -6.5 % | 41.1 -> 24.3 | INNOVATION |

Verdict by the registered bar: does not survive (no co-primary passes, no interval above zero, so not "lean" either). The reference
answers why: the Binance model restricted to the same coverage falls just as far (+0.038, NEUTRAL), and the Kraken model is level
with or above it on every measure (d Sharpe +0.065 vs +0.038, same next-day IC, every point estimate above all 20 placebos). The loss
comes from coverage (9 of 30 coins without a forecast, October 2026 empty), not from the training data: Kraken bars carry the
ranking effect. A side finding: the Kraken forecast does not correlate with the trend score (+0.004 against -0.13).

## 4. Recommendation (owner decision; not a legal opinion)

1. On the text, Kraken is the only free route with no ML or trading-use clause; its grant is "only for your own benefit", with no
   third-party or commercial exploitation and broad anti-automation wording. If the owner (or counsel) reads bot sizing of the
   owner's own account as "own benefit", the Kraken-trained weights are the candidate: they hold the forecast quality, and they
   stay private (no publication, no paid API).
2. Coverage is what limits it. A Kraken-trained model can only forecast coins that Kraken lists, from Kraken's own bars, unless the
   live inputs for the other coins come from a source whose terms allow that use. That is the same licence question again: live
   Binance API bars are under Binance's API terms (not read here).
3. For full coverage under terms that state trading and own-model training: Tardis.dev Solo spot (USD 900 / month, 4 years of
   history on yearly billing; 9.5 allows own models, internal use only, Coinbase data excluded) covers Binance spot; but Solo's 4 years reach back only to about 2022-11 (5 years when invoiced), so the
   2019-2022 training span of the current recipe needs the Business tier (USD 3,500 / month, "All available") or a shortened
   training window (untested). It is the cleanest paid route. CoinGecko Analyst (USD 129 / month, commercial licence,
   hourly from 2018) is cheaper but has close / volume only; it needs a `mkt+r24` variant without the range inputs (untested).
4. Not usable for this purpose on their text: Coinbase (ML barred), CryptoCompare / CoinDesk (non-commercial, "not intended for
   trading", free tier retired), Bitstamp (personal, no automated collection), Alpaca (personal and noncommercial; also too few
   symbols), Gemini (no history; commercial use by agreement).

Artifacts (private, git-ignored): `runs-agent/lic/` (cache, universe, checkpoints `r24/ckpt/mkt+r24_s{0,1}_F*.pt`, forecasts,
gate JSONs `gate/gate_gt.json`, `gate/gate_gtr.json`, `gate_ref/gate_gt.json`, run scripts `run.sh`, `gate.sh`).

## 5. Test: `mkt+r24` on the owner's venue API bars, KuCoin 1 h klines (`agent/PREREG-LIC2.md`, wave 4e)

Source: KuCoin public REST `GET /api/v1/market/candles?type=1hour`, USDT pairs, no key. 808 USDT pairs in the list, 780 after the
registered exclusions; 9,204 requests in 43 minutes on 4 kept-alive connections (3.6 requests/s, capped at 5), no 429 answer;
completed bars to 2026-11-18 09:00 UTC; cache `runs-agent/lic2/cache` (never published). 232 pairs have a bar before 2023 and all of
them pass the liquidity filter (a bar in every hour of the 2022 window): pool 232, evaluation 64, held-out 16, train 216; listed by
January 2020 / 2021 / 2022: 31 / 67 / 187. The endpoint serves nothing for pairs KuCoin delisted, so the pool is survivors only. One
stablecoin that the fixed exclusion lists do not name (USDD) is in the pool. Same recipe as waves 4 and 4d, seeds 0 and 1, 333 s +
330 s GPU under the 3.2 GB cap, no deviation from the recipe. OKX `history-candles` was used for three coins only (QTUM, RVN, ZRX;
2,148 requests).

Data check, 2023-01 .. 2026-10, pool symbols present in both caches (reported; no pair was dropped because of it):

| against | shared | hourly-return correlation: median (10th percentile, lowest) | its bar hours missing in KuCoin | KuCoin bar hours missing in it (median) | price gap (median) |
|---|---|---|---|---|---|
| Binance cache | 146 | 0.969 (0.914, 0.741); harness coins lowest ONE 0.826, IOST 0.838 | none | 0 % (up to 70 % for one symbol) | 9 bp |
| Kraken cache | 138 | 0.881 (0.683, -0.08 for USDD); harness coins lowest NEO 0.598 | none | 15 % | 20 bp |
| OKX fill bars against Binance | 3 | QTUM 0.985, ZRX 0.983, RVN 0.971 | (Binance bar hours missing in OKX: none) | | 6-7 bp |

Coverage (registered report; Binance forecast file = 42,000 R9-R24 cells, 50,341 ALGOS cells):

| forecast | R9-R24 coins of 30 | from 2023-01-01 | cells | October 2026 | ALGOS coins of 36 | cells |
|---|---|---|---|---|---|---|
| primary: KuCoin-trained, KuCoin bars | 27 (no QTUM, RVN, ZRX) | 27 | 37,800 (90.0 %) | yes | 33 (no ICX, RVN, ZRX) | 46,200 (91.8 %) |
| S1: Kraken weights, Kraken bars + KuCoin bars for the coins outside the Kraken pool | 29 (no RVN) | 26 (BAND, COMP one day, ZRX two days later) | 37,805 (90.0 %) | no (bars end 2026-09-30) | 35 (no RVN) | 45,928 (91.2 %) |
| S2: Kraken weights, KuCoin bars only | 29 (no RVN; ZRX from 2024-08-06, QTUM from 2025-07-27) | 27 | 39,079 (93.0 %) | yes | 34 (no ICX, RVN) | 47,017 (93.4 %) |
| S3: primary + OKX bars for QTUM, RVN, ZRX | 30 | 30 | 42,000 (100 %) | yes | 35 (no ICX) | 49,000 (97.3 %) |
| wave 4d: Kraken weights, Kraken bars | 21 | 18 | 26,853 (63.9 %) | no | 26 | 33,607 (66.8 %) |

Why KuCoin misses three R9-R24 coins: RVN-USDT (and ICX-USDT) are delisted and the endpoint serves no bars for them; KuCoin's
QTUM-USDT and ZRX-USDT pairs start in 2024-11 and 2023-11, after the pool cut-off (first bar by 2022-12). Late but early enough:
HBAR and BAND 2021-09, IOTA 2021-12, TFUEL 2022-03. In S1 the 8 coins filled from KuCoin bars are HBAR, IOTA, NEO, ONE, TFUEL, VET,
ZEN, ZIL (QTUM and ZRX are in the Kraken pool).

Forecast quality:

| measure | primary: KuCoin weights, KuCoin bars | Binance `mkt+r24`, the primary's cells | S1: Kraken weights, Kraken + KuCoin bars | S2: Kraken weights, KuCoin bars |
|---|---|---|---|---|
| validation 24 h rank IC, KuCoin seen symbols (48), 8,544 origins | +0.091 [+0.080, +0.103]; registered Binance `ohlc+cal` on the same rows +0.079, d +0.012 [+0.003, +0.022] | | | |
| next-day IC on the ALGOS harness (14 universes, 19,484 universe-days) | +0.043 [+0.031, +0.056] | +0.069 [+0.057, +0.082] | +0.065 [+0.052, +0.078] | +0.044 [+0.032, +0.056] |
| rho with the trend score | -0.082 [-0.098, -0.065] | -0.134 [-0.153, -0.115] | -0.024 [-0.047, -0.002] | -0.026 [-0.045, -0.007] |
| per-day rank correlation with the Binance forecast, R9-R24 cells | 0.70 (ALGOS cells 0.70) | | 0.69 | 0.69 |

Tilt on R9-R24 (2023-01 .. 2026-10, form `gt` unless marked, 20 placebos each, base arms from a private copy of `runs-agent/r24/confirm`):

| forecast / form | cost | mean d Sharpe [95 %] | share | years + (2023-26) | universes + | placebo p95 / max | turnover | ruin dd60 (M11) | class |
|---|---|---|---|---|---|---|---|---|---|
| primary `gt` (co-primary) | c7 | +0.069 [-0.047, +0.186] | 68 % | 3/4 (-0.05 +0.07 +0.14 +0.18) | 11/16 | +0.019 / +0.022 | -4.7 % | 41.1 -> 25.1 | NEUTRAL |
| primary `gt` | c10 | +0.066 [-0.050, +0.183] | 67 % | 3/4 | 11/16 | +0.016 / +0.019 | -4.7 % | 41.2 -> 25.4 | NEUTRAL |
| primary `gtr` risk-matched (co-primary) | c7 | +0.013 [-0.067, +0.100] | 55 % | 3/4 (-0.10 +0.03 +0.04 +0.11) | 10/16 | -0.018 / -0.012 | +3.5 % | 41.1 -> 36.8 | NEUTRAL |
| primary `gtr` | c10 | +0.008 [-0.072, +0.096] | 54 % | 3/4 | 10/16 | -0.022 / -0.016 | +3.5 % | 41.2 -> 37.0 | NEUTRAL |
| S1: Kraken weights, Kraken bars + KuCoin bars for 8 coins (29/30, no October) | c7 | +0.143 [+0.022, +0.260] | 80.3 % | 4/4 (+0.11 +0.17 +0.19 +0.15) | 14/16 | +0.007 / +0.011 | -4.1 % | 41.1 -> 27.3 | INNOVATION |
| S1 | c10 | +0.140 [+0.019, +0.257] | 79.7 % | 4/4 | 14/16 | +0.004 / +0.008 | -4.1 % | 41.2 -> 27.8 | NEUTRAL (+) |
| S2: Kraken weights, KuCoin bars only (29/30) | c7 | +0.098 [-0.011, +0.209] | 73 % | 4/4 (+0.01 +0.10 +0.14 +0.18) | 13/16 | +0.015 / +0.015 | -4.0 % | 41.1 -> 28.5 | NEUTRAL |
| S2 | c10 | +0.095 [-0.014, +0.206] | 73 % | 4/4 | 13/16 | +0.013 / +0.013 | -4.0 % | 41.2 -> 28.6 | NEUTRAL |
| S3: primary + OKX bars for QTUM, RVN, ZRX (30/30) | c7 | +0.093 [-0.024, +0.209] | 71 % | 3/4 (-0.03 +0.09 +0.18 +0.20) | 12/16 | +0.034 / +0.038 | -4.9 % | 41.1 -> 25.1 | NEUTRAL |
| S3 | c10 | +0.090 [-0.028, +0.207] | 70 % | 3/4 | 12/16 | +0.031 / +0.035 | -4.9 % | 41.2 -> 25.4 | NEUTRAL |
| reference: Binance `mkt+r24`, the primary's cells (27/30) | c7 | +0.120 [+0.005, +0.241] | 78 % | 3/4 (-0.03 +0.17 +0.18 +0.22) | 15/16 | +0.018 / +0.019 | -6.0 % | 41.1 -> 24.3 | NEUTRAL (+) |
| reference | c10 | +0.116 [+0.001, +0.238] | 77 % | 3/4 | 15/16 | +0.015 / +0.016 | -6.0 % | 41.2 -> 24.6 | NEUTRAL (+) |
| (wave 4c) Binance `mkt+r24`, full coverage | c7 | +0.141 [+0.024, +0.263] | 81 % | 4/4 | 15/16 | +0.028 / +0.029 | -6.5 % | 41.1 -> 24.3 | INNOVATION |
| (wave 4d) Kraken weights, Kraken bars (21/30) | c7 | +0.065 [-0.049, +0.180] | 68 % | 4/4 | 12/16 | -0.016 / -0.006 | -3.0 % | 41.1 -> 28.6 | NEUTRAL |

Verdict by the registered bar: does not survive. Neither co-primary passes and neither has its interval above zero, so it is not
"lean" either; `gtr` also adds +3.5 % turnover. No secondary meets the bar. S1 is INNOVATION at c7 and misses the class at c10 on
the share line (79.7 % against 80 %), with the interval above zero at both costs: "lean" in the bar's terms, and a secondary in any
case, so it needs its own confirmation. Every row is above all 20 of its placebos.

What the rows say:
- Coverage is not the main loss this time. The Binance model on the primary's 27 coins keeps +0.120 [+0.005, +0.241] and 15/16
  universes (the three missing coins cost +0.141 -> +0.120 and the class, share 78 %); the KuCoin forecast on the same cells gets
  +0.069. Filling the three coins from OKX (S3, 30/30) gives +0.093, still NEUTRAL. The registered reading had only two cases
  (reference fails = coverage, reference passes = model); the reference landed between them, so both parts are reported.
- The loss follows the bars the forecast is computed from, not the bars it was trained on. On the ALGOS harness the next-day IC is
  +0.043 for KuCoin weights on KuCoin bars and +0.044 for Kraken weights on KuCoin bars, against +0.069 for the Binance model on the
  same cells and +0.065 for Kraken weights on (mostly) Kraken bars. About a third of the ranking information is lost when the inputs
  are KuCoin bars. KuCoin's own validation IC is high (+0.091), so the model does rank KuCoin's next-day returns; the harness books
  are priced on Binance. KuCoin hourly returns track Binance at 0.97 (median), 0.83 for the thinnest harness coin. Why that gap is
  enough to lose a third of the IC was not tested.
- Kraken weights with KuCoin bars only for the coins Kraken lacks (S1) reach the level of the Binance model at full coverage:
  +0.143 against +0.141, 14/16 universes, 4/4 years, without October 2026 and without RVN. This agrees with wave 4d's reading that
  its NEUTRAL was coverage. It is one of four further looks in this wave (S1, S2, S3, reference), on universes now used in three
  waves.

After October 2026 (registered consistency read, no interval claimed): KuCoin-trained forecast, F4 checkpoints, 16 origins
2026-11-01 .. 2026-11-16, 41 coins, per-day IC against the next 24 h KuCoin return: mean +0.127, 69 % of days above zero; the same
statistic over 2023-01 .. 2026-10 (1,400 days) is +0.057 and 60 %. Same sign; nothing more can be said from 16 days.

Notes on the run: the first-bar request needed an explicit `startAt` (found on BTC-USDT before any bar was stored); the first fetch
was stopped after about two minutes and resumed with kept-alive connections (same snapshot hour); RVN was fetched from OKX before
the cache was built, QTUM and ZRX after it showed them outside the pool; the gate resume caches (21 GB) were deleted after the JSONs
were written.

## 6. Recommendation after wave 4e (owner decision; not a legal opinion)

1. Training and inferring on KuCoin API bars does not reproduce the tilt. It is not an integration candidate, whatever the reading
   of KuCoin's terms.
2. The candidate that the numbers point to is S1's set-up: the Kraken-trained weights (section 4, point 1), Kraken bars as inputs
   for the coins Kraken lists and for the market series, and KuCoin API bars only for the 8 R9-R24 coins Kraken lacks. It uses no
   KuCoin or OKX bar for training and no OKX bar at all; KuCoin bars enter as inference inputs only. It is not confirmed: it
   needs a registered test on universes or dates it has not touched (the February 2027 forward scoring, or new universes).
3. What a live job would need as the inference bar source for that set-up: (a) hourly Kraken bars for the 144 training-pool
   symbols (market mean, dispersion, breadth) and the 21 Kraken-listed forecast coins, with 6,200 h of history per forecast coin.
   Kraken's public `OHLC` endpoint returns only the most recent 720 hourly bars (checked 2026-11-18: one request for XBTUSD came
   back from 2026-10-19), so the cache has to be seeded from the OHLCVT archive (it ends 2026-09-30; the next quarterly file or the
   public `Trades` endpoint has to close the gap) and topped up at least every 30 days. (b) KuCoin `GET /api/v1/market/candles?type=1hour` for HBAR, IOTA, NEO, ONE, TFUEL, VET, ZEN, ZIL: five
   requests per coin to seed 6,200 h, one request per coin per day afterwards. (c) Only completed bars; the forecast uses the
   23:00 UTC bar and is applied at 00:00 UTC. (d) RVN is outside the Kraken pool and KuCoin serves no bars for it; it stays untilted
   (multiplier 1) or takes OKX bars (S3: the three-coin OKX fill moved the point estimate from +0.069 to +0.093, inside the noise).
4. The terms text for the two venues is in section 1 and under "Quotes and URLs": KuCoin has no API, trading-use or
   machine-learning clause and reserves the content of the Platform against commercial use and systematic collection; OKX limits
   Market Data to "your own personal, non-commercial trading and account management purposes" and bars training only for models
   "intended to be commercialized or made available to third parties".

Artifacts (private, git-ignored): `runs-agent/lic2/` (KuCoin download `dl/`, cache, universe, checkpoints
`r24/ckpt/mkt+r24_s{0,1}_F*.pt`, forecasts `r24/botpred*.npz`, `s1/`, `s2/`, `s3/`, `ref/`, gate JSONs `gate/gate_gt.json`,
`gate/gate_gtr.json`, `gate_s{1,2,3}/gate_gt.json`, `gate_ref/gate_gt.json`, `algos_*/corr.json`, `sanity_*.json`, `val.json`,
scripts `run.sh`, `gate.sh`, `corr.sh`).
