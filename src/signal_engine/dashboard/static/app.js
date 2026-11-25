/**
 * signal-engine | Dashboard Frontend Logic
 * Coordinates REST API fetching, DOM population, modal triggers,
 * and Chart.js rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── Elements & State ─────────────────────────────────────
    let equityChart = null;
    let modalPriceChart = null;
    let scanResultsCache = []; // Store reports for modal details

    const state = {
        currentTab: "scan",
        scanDate: new Date().toISOString().split("T")[0]
    };

    // Set default date input value to today
    const dateInput = document.getElementById("scan-date-input");
    if (dateInput) {
        dateInput.value = state.scanDate;
        dateInput.addEventListener("change", (e) => {
            state.scanDate = e.target.value;
        });
    }

    // ── Tab Navigation ───────────────────────────────────────
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabViews = document.querySelectorAll(".tab-view");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        state.currentTab = tabId;

        // Toggle nav buttons active class
        navButtons.forEach(btn => {
            if (btn.getAttribute("data-tab") === tabId) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        // Toggle subviews
        tabViews.forEach(view => {
            if (view.id === `tab-${tabId}`) {
                view.classList.add("active");
            } else {
                view.classList.remove("active");
            }
        });

        // Load data on activation
        if (tabId === "backtest") {
            loadBacktestData();
        } else if (tabId === "paper") {
            loadPaperData();
        } else if (tabId === "leaps") {
            loadLEAPSData();
        } else if (tabId === "traders") {
            loadTradersData();
        } else if (tabId === "weekend") {
            loadWeekendTabData();
            startWeekendPolling();
        }

        if (tabId !== "weekend") {
            stopWeekendPolling();
        }
    }

    // ── System Status Ingestion ──────────────────────────────
    async function checkSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            const modeDot = document.getElementById("status-mode-dot");
            const modeText = document.getElementById("status-mode-text");
            const simBanner = document.getElementById("simulated-banner");

            if (data.mode === "SIMULATED_DEMO") {
                modeDot.className = "status-dot simulated";
                modeText.textContent = "SIMULATED DEMO";
                modeText.style.color = "var(--color-hold)";
                if (simBanner) simBanner.style.display = "flex";
            } else {
                modeDot.className = "status-dot online";
                modeText.textContent = "LIVE DATA (CONNECTED)";
                modeText.style.color = "var(--color-online)";
                if (simBanner) simBanner.style.display = "none";
            }

            // Set individual API badges
            toggleBadge("badge-alpaca", data.alpaca_connected);
            toggleBadge("badge-finnhub", data.finnhub_connected);
            toggleBadge("badge-newsapi", data.newsapi_connected);
            toggleBadge("badge-alpaca-paper", data.alpaca_paper_connected);

        } catch (err) {
            console.error("Status check failed:", err);
        }
    }

    function toggleBadge(id, isConnected) {
        const badge = document.getElementById(id);
        if (badge) {
            if (isConnected) {
                badge.className = "api-badge online";
            } else {
                badge.className = "api-badge offline";
            }
        }
    }

    // Initialize status
    checkSystemStatus();

    // ── Tab 1: Scan Logic ────────────────────────────────────
    const btnRunScan = document.getElementById("btn-run-scan");
    const scanTableBody = document.getElementById("scan-table-body");
    const scanCountLabel = document.getElementById("scan-count-label");

    if (btnRunScan) {
        btnRunScan.addEventListener("click", runSignalScan);
    }

    const REC_META = {
        BUY: { icon: "fa-arrow-trend-up", label: "BUY", cls: "buy" },
        SHORT: { icon: "fa-arrow-trend-down", label: "SHORT", cls: "short" },
        HOLD: { icon: "fa-minus", label: "HOLD", cls: "hold" },
    };

    function recPillHtml(recommendation) {
        const meta = REC_META[recommendation] || REC_META.HOLD;
        return `<span class="rec-pill rec-pill-lg ${meta.cls}"><i class="fa-solid ${meta.icon}"></i> ${meta.label}</span>`;
    }

    function patternChipsHtml(matchedPatterns) {
        const names = (matchedPatterns || "").split(",").map(s => s.trim()).filter(Boolean);
        if (!names.length) return `<span class="text-muted">—</span>`;
        return names.map(n => `<span class="pattern-chip">${n}</span>`).join(" ");
    }

    async function runSignalScan() {
        // Show loading state
        scanTableBody.innerHTML = `
            <tr>
                <td colspan="11" class="text-center py-5">
                    <i class="fa-solid fa-circle-notch fa-spin fa-2x text-muted mb-2"></i>
                    <p>Executing analyzers & composite scorers... Please wait.</p>
                </td>
            </tr>
        `;

        try {
            const res = await fetch(`/api/scan?date=${state.scanDate}`);
            const data = await res.json();

            scanResultsCache = data.results || [];
            if (scanCountLabel) {
                scanCountLabel.textContent = `${scanResultsCache.length} tickers loaded (${data.mode || 'Demo'} mode)`;
            }

            if (!scanResultsCache.length) {
                scanTableBody.innerHTML = `<tr><td colspan="11" class="text-center py-4">No signals returned. Verify watchlist configuration.</td></tr>`;
                return;
            }

            scanTableBody.innerHTML = "";
            scanResultsCache.forEach((row, index) => {
                const tr = document.createElement("tr");
                tr.setAttribute("data-ticker", row.ticker);
                tr.classList.add(`row-${(REC_META[row.recommendation] || REC_META.HOLD).cls}`);

                // Color codes
                const scoreClass = row.composite_score > 0.3 ? "positive" : (row.composite_score < -0.3 ? "negative" : "");
                const patternScore = row.pattern_score || 0;
                const pattScoreClass = patternScore > 0 ? "positive" : (patternScore < 0 ? "negative" : "");

                // Primary rule trigger extraction
                const rulesList = (row.triggered_rules || "").split(" | ");
                const primaryRule = rulesList[0] || "No major triggers";

                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td><strong>${row.ticker}</strong></td>
                    <td>${recPillHtml(row.recommendation)}</td>
                    <td><span class="score-text ${scoreClass}">${row.composite_score > 0 ? '+' : ''}${row.composite_score.toFixed(3)}</span></td>
                    <td><span class="conf-badge ${row.confidence.toLowerCase()}">${row.confidence}</span></td>
                    <td>${patternChipsHtml(row.matched_patterns)}</td>
                    <td>${row.technical_score > 0 ? '+' : ''}${row.technical_score.toFixed(2)}</td>
                    <td>${row.sentiment_score > 0 ? '+' : ''}${row.sentiment_score.toFixed(2)}</td>
                    <td>${row.political_score > 0 ? '+' : ''}${row.political_score.toFixed(2)}</td>
                    <td><span class="score-text ${pattScoreClass}">${patternScore > 0 ? '+' : ''}${patternScore.toFixed(2)}</span></td>
                    <td class="text-muted"><small>${primaryRule}</small></td>
                `;

                // Modal row click
                tr.addEventListener("click", () => {
                    openDetailModal(row);
                });

                scanTableBody.appendChild(tr);
            });

        } catch (err) {
            console.error("Scan fetch failed:", err);
            scanTableBody.innerHTML = `<tr><td colspan="11" class="text-center py-4 red-text"><i class="fa-solid fa-circle-exclamation"></i> Error running scan: ${err.message}</td></tr>`;
        }
    }

    // Auto-run scan on load
    runSignalScan();

    // ── Tab 2: Backtest Logic ─────────────────────────────────
    const btnRunBacktest = document.getElementById("btn-run-backtest");
    if (btnRunBacktest) {
        btnRunBacktest.addEventListener("click", loadBacktestData);
    }

    async function loadBacktestData() {
        setBacktestLoadingState();

        try {
            const res = await fetch("/api/backtest");
            const data = await res.json();

            // Populate cards
            const ret = data.metrics.total_return_pct;
            const win = data.metrics.win_rate_pct;
            const sharpe = data.metrics.sharpe_ratio;
            const dd = data.metrics.max_drawdown_pct;

            document.getElementById("bt-metric-return").textContent = `${ret > 0 ? '+' : ''}${ret.toFixed(2)}%`;
            document.getElementById("bt-metric-return").className = ret >= 0 ? "green-text" : "red-text";

            document.getElementById("bt-metric-winrate").textContent = `${win.toFixed(1)}%`;
            
            const winBadge = document.getElementById("bt-winrate-badge");
            const winIcon = document.getElementById("bt-winrate-icon");
            if (win >= 70) {
                winBadge.textContent = "✅ TARGET MET";
                winBadge.className = "badge badge-info"; // green indicator
                winBadge.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
                winBadge.style.color = "var(--color-buy)";
                winIcon.style.color = "var(--color-buy)";
                winIcon.style.background = "rgba(16, 185, 129, 0.08)";
            } else {
                winBadge.textContent = "❌ LOW WINRATE";
                winBadge.className = "badge";
                winBadge.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
                winBadge.style.color = "var(--color-sell)";
                winIcon.style.color = "var(--color-sell)";
                winIcon.style.background = "rgba(239, 68, 68, 0.08)";
            }

            document.getElementById("bt-metric-sharpe").textContent = sharpe.toFixed(2);
            document.getElementById("bt-metric-drawdown").textContent = `${dd.toFixed(2)}%`;

            // Details panel
            document.getElementById("bt-detail-trades").textContent = data.metrics.total_trades;
            document.getElementById("bt-detail-wins").textContent = data.metrics.winning_trades;
            document.getElementById("bt-detail-losses").textContent = data.metrics.losing_trades;
            document.getElementById("bt-detail-dir-acc").textContent = `${data.metrics.directional_accuracy_pct.toFixed(1)}%`;

            // Render Chart
            renderEquityChart(data.equity_curve);

        } catch (err) {
            console.error("Backtest loading failed:", err);
            setBacktestErrorState(err.message);
        }
    }

    function setBacktestLoadingState() {
        document.getElementById("bt-metric-return").textContent = "...";
        document.getElementById("bt-metric-winrate").textContent = "...";
        document.getElementById("bt-metric-sharpe").textContent = "...";
        document.getElementById("bt-metric-drawdown").textContent = "...";
        document.getElementById("bt-detail-trades").textContent = "...";
        document.getElementById("bt-detail-wins").textContent = "...";
        document.getElementById("bt-detail-losses").textContent = "...";
        document.getElementById("bt-detail-dir-acc").textContent = "...";
    }

    function setBacktestErrorState(msg) {
        document.getElementById("bt-metric-return").textContent = "ERR";
        document.getElementById("bt-metric-winrate").textContent = "ERR";
    }

    function renderEquityChart(curve) {
        const ctx = document.getElementById("equity-chart").getContext("2d");
        
        if (equityChart) {
            equityChart.destroy();
        }

        const labels = curve.map(p => p.date);
        const values = curve.map(p => p.value);

        // Chart gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, "rgba(139, 92, 246, 0.35)");
        gradient.addColorStop(1, "rgba(139, 92, 246, 0.0)");

        equityChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Portfolio Value ($)",
                    data: values,
                    borderColor: "#8b5cf6",
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "#8b5cf6",
                    fill: true,
                    backgroundColor: gradient,
                    tension: 0.15
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#9ca3af", font: { family: "Inter", size: 10 } }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#9ca3af", font: { family: "Inter", size: 10 } }
                    }
                }
            }
        });
    }

    // ── Tab 3: Paper Portfolio Logic ──────────────────────────
    async function loadPaperData() {
        try {
            const res = await fetch("/api/paper");
            const data = await res.json();

            // Account metrics
            document.getElementById("paper-portfolio-value").textContent = `$${data.account.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById("paper-buying-power").textContent = `$${data.account.buying_power.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById("paper-cash").textContent = `$${data.account.cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

            // Logged Open Trades table (SQLite)
            const openBody = document.getElementById("paper-open-body");
            if (openBody) {
                if (!data.logged_open_trades || !data.logged_open_trades.length) {
                    openBody.innerHTML = `<tr><td colspan="7" class="text-center py-4">No open or queued paper trades.</td></tr>`;
                } else {
                    openBody.innerHTML = "";
                    data.logged_open_trades.forEach(trade => {
                        const tr = document.createElement("tr");
                        const notionalStr = trade.notional
                            ? `$${trade.notional.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
                            : "N/A";
                        const scoreStr = trade.composite_score ? trade.composite_score.toFixed(3) : "N/A";

                        tr.innerHTML = `
                            <td><strong>${trade.ticker}</strong></td>
                            <td><span class="rec-pill buy" style="padding: 2px 8px;">${trade.recommendation || 'BUY'}</span></td>
                            <td>${(trade.entry_date || "").slice(0, 10)}</td>
                            <td style="font-weight: 600;">${notionalStr}</td>
                            <td><span class="score-text positive">${scoreStr}</span></td>
                            <td><span class="badge badge-info">${trade.hold_period || '1_year'}</span></td>
                            <td><span class="status-dot online" style="display:inline-block; margin-right:4px;"></span> <span style="color: var(--color-buy);">QUEUED / ACCEPTED</span></td>
                        `;
                        openBody.appendChild(tr);
                    });
                }
            }

            // Live Positions table (Alpaca Broker filled positions)
            const posBody = document.getElementById("paper-positions-body");
            if (!data.positions || !data.positions.length) {
                posBody.innerHTML = `<tr><td colspan="7" class="text-center py-4">No filled open positions in sandbox (market closed — orders queued).</td></tr>`;
            } else {
                posBody.innerHTML = "";
                data.positions.forEach(pos => {
                    const tr = document.createElement("tr");
                    const pl = pos.unrealized_pl || 0;
                    const plpc = (pos.unrealized_plpc || 0) * 100;
                    const plStyle = pl >= 0 ? "green-text" : "red-text";

                    tr.innerHTML = `
                        <td><strong>${pos.symbol}</strong></td>
                        <td>${pos.qty}</td>
                        <td>$${pos.avg_entry_price.toFixed(2)}</td>
                        <td>$${(pos.current_price || 0).toFixed(2)}</td>
                        <td>$${(pos.market_value || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                        <td class="${plStyle}">$${pl >= 0 ? '+' : ''}${pl.toFixed(2)}</td>
                        <td class="${plStyle}">${plpc >= 0 ? '+' : ''}${plpc.toFixed(2)}%</td>
                    `;
                    posBody.appendChild(tr);
                });
            }

            // Closed Trades table
            const closedBody = document.getElementById("paper-closed-body");
            const winrateBadge = document.getElementById("paper-winrate-badge");

            if (data.performance_summary.total_closed_trades > 0) {
                const wr = data.performance_summary.win_rate_pct;
                winrateBadge.textContent = `Win Rate: ${wr.toFixed(1)}%`;
                winrateBadge.style.backgroundColor = wr >= 70 ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)";
                winrateBadge.style.color = wr >= 70 ? "var(--color-buy)" : "var(--color-sell)";
            } else {
                winrateBadge.textContent = "No closed trades";
            }

            if (!data.logged_closed_trades || !data.logged_closed_trades.length) {
                closedBody.innerHTML = `<tr><td colspan="7" class="text-center py-4">No closed trade history logs found in SQLite.</td></tr>`;
            } else {
                closedBody.innerHTML = "";
                data.logged_closed_trades.forEach(trade => {
                    const tr = document.createElement("tr");
                    const pl = trade.outcome_pct || 0;
                    const plStyle = pl > 0 ? "green-text" : "red-text";

                    tr.innerHTML = `
                        <td><strong>${trade.ticker}</strong></td>
                        <td>${(trade.entry_date || "").slice(0, 10)}</td>
                        <td>${(trade.exit_date || "").slice(0, 10)}</td>
                        <td>$${(trade.entry_price || 0).toFixed(2)}</td>
                        <td>$${(trade.exit_price || 0).toFixed(2)}</td>
                        <td>$${(trade.notional || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                        <td class="${plStyle}">${pl > 0 ? '+' : ''}${pl.toFixed(2)}%</td>
                    `;
                    closedBody.appendChild(tr);
                });
            }

        } catch (err) {
            console.error("Paper tab load failed:", err);
        }
    }

    // ── Tab 4: LEAPS screener Logic ────────────────────────────
    async function loadLEAPSData() {
        const body = document.getElementById("leaps-table-body");
        body.innerHTML = `
            <tr>
                <td colspan="12" class="text-center py-4">
                    <i class="fa-solid fa-circle-notch fa-spin text-muted mb-2"></i>
                    <p>Screening candidates &amp; running Black-Scholes estimates...</p>
                </td>
            </tr>
        `;

        try {
            const res = await fetch("/api/leaps");
            const data = await res.json();

            if (!data.candidates || !data.candidates.length) {
                body.innerHTML = `<tr><td colspan="12" class="text-center py-4">No options candidates found.</td></tr>`;
                return;
            }

            body.innerHTML = "";
            data.candidates.forEach((cand, index) => {
                const tr = document.createElement("tr");
                const scoreClass = cand.leaps_score >= 0.6 ? "positive" : (cand.leaps_score < 0.3 ? "negative" : "");

                // 1yr target
                const targetStr = cand.target_price_1yr ? `$${cand.target_price_1yr.toFixed(2)}` : "N/A";
                const upsideStr = cand.target_upside_pct
                    ? `<span class="positive">+${cand.target_upside_pct.toFixed(1)}%</span>`
                    : "N/A";

                // Vol
                const volStr = cand.hist_volatility ? `${(cand.hist_volatility * 100).toFixed(0)}%` : "N/A";

                // Premium per share
                const premStr = cand.est_call_premium ? `$${cand.est_call_premium.toFixed(2)}` : "N/A";

                // Est return on $1k
                let returnStr = "N/A";
                if (cand.est_return_on_1k !== null && cand.est_return_on_1k !== undefined
                    && cand.est_return_pct_on_1k !== null && cand.est_return_pct_on_1k !== undefined) {
                    const retClass = cand.est_return_on_1k >= 0 ? "positive" : "negative";
                    const sign = cand.est_return_on_1k >= 0 ? "+" : "";
                    const absRet = Math.abs(cand.est_return_on_1k);
                    const absPct = Math.abs(cand.est_return_pct_on_1k);
                    returnStr = `<span class="${retClass}" style="font-weight:600;">${sign}$${absRet.toLocaleString(undefined, {maximumFractionDigits: 0})} (${sign}${absPct.toFixed(0)}%)</span>`;
                }

                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td><strong>${cand.ticker}</strong></td>
                    <td><span style="color: var(--color-buy); font-weight: 600; font-size: 12px;">📈 CALL</span></td>
                    <td><span class="score-text ${scoreClass}">${cand.leaps_score.toFixed(3)}</span></td>
                    <td>$${(cand.current_price || 0).toFixed(2)}</td>
                    <td>$${(cand.suggested_strike || 0).toFixed(2)}</td>
                    <td style="font-weight: 600;">${targetStr}</td>
                    <td>${upsideStr}</td>
                    <td style="color: var(--text-muted);">${volStr}</td>
                    <td style="color: var(--text-muted);">${premStr}</td>
                    <td>${returnStr}</td>
                    <td><small>${(cand.catalysts || []).slice(0, 2).join(" | ")}</small></td>
                `;
                body.appendChild(tr);
            });

        } catch (err) {
            console.error("LEAPS screen failed:", err);
            body.innerHTML = `<tr><td colspan="12" class="text-center py-4 red-text">Screening failed: ${err.message}</td></tr>`;
        }
    }

    // ── Modal Drawer Detail View ──────────────────────────────
    const detailModal = document.getElementById("detail-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");

    if (btnCloseModal) {
        btnCloseModal.addEventListener("click", () => {
            detailModal.classList.remove("active");
        });
    }

    // Close on overlay click
    if (detailModal) {
        detailModal.addEventListener("click", (e) => {
            if (e.target === detailModal) {
                detailModal.classList.remove("active");
            }
        });
    }

    function openDetailModal(row) {
        if (!detailModal) return;

        // Set basic values
        document.getElementById("modal-ticker").textContent = row.ticker;
        document.getElementById("modal-name").textContent = row.name || row.ticker;

        const compositeText = document.getElementById("modal-composite-score");
        const compVal = row.composite_score;
        compositeText.textContent = `${compVal >= 0 ? '+' : ''}${compVal.toFixed(3)}`;
        
        // Colors
        if (compVal >= 0.3) {
            compositeText.className = "green-text";
        } else if (compVal <= -0.3) {
            compositeText.className = "red-text";
        } else {
            compositeText.className = "";
        }

        const recPill = document.getElementById("modal-recommendation");
        const recMeta = REC_META[row.recommendation] || REC_META.HOLD;
        recPill.innerHTML = `<i class="fa-solid ${recMeta.icon}"></i> ${recMeta.label}`;
        recPill.className = `rec-pill rec-pill-lg ${recMeta.cls}`;

        document.getElementById("modal-confidence").textContent = `${row.confidence} CONFIDENCE`;

        // Component scores & Progress bars
        setScoreBar("modal-score-tech", "modal-fill-tech", row.technical_score);
        setScoreBar("modal-score-sent", "modal-fill-sent", row.sentiment_score);
        setScoreBar("modal-score-pol", "modal-fill-pol", row.political_score);
        setScoreBar("modal-score-pattern", "modal-fill-pattern", row.pattern_score || 0);

        // Rules List
        const rulesList = document.getElementById("modal-rules-list");
        rulesList.innerHTML = "";
        const rules = (row.triggered_rules || "").split(" | ");
        rules.forEach(rule => {
            const li = document.createElement("li");
            let icon = `<i class="fa-solid fa-circle-info"></i>`;
            if (rule.startsWith("✅")) {
                icon = `<i class="fa-solid fa-circle-check green-text"></i>`;
                rule = rule.replace("✅", "");
            } else if (rule.startsWith("🔴")) {
                icon = `<i class="fa-solid fa-circle-exclamation red-text"></i>`;
                rule = rule.replace("🔴", "");
            } else if (rule.startsWith("⚠️")) {
                icon = `<i class="fa-solid fa-triangle-exclamation" style="color: var(--color-hold);"></i>`;
                rule = rule.replace("⚠️", "");
            } else if (rule.startsWith("📊")) {
                icon = `<i class="fa-solid fa-chart-simple text-muted"></i>`;
                rule = rule.replace("📊", "");
            }
            li.innerHTML = `${icon} <span>${rule.trim()}</span>`;
            rulesList.appendChild(li);
        });

        // News Headlines list
        const headlinesList = document.getElementById("modal-headlines-list");
        headlinesList.innerHTML = "";
        const headlines = (row.top_headlines || "").split(" | ");
        headlines.forEach(headline => {
            if (headline.trim()) {
                const li = document.createElement("li");
                // Check if political keyword matched is populated
                const kw = row.political_keywords ? `<span class="badge" style="background: rgba(139,92,246,0.1); color: var(--accent-primary); margin-left: 8px;">Matched: ${row.political_keywords}</span>` : "";
                
                li.innerHTML = `
                    <div class="headline-meta">Relevant Source ${kw}</div>
                    <p class="headline-text">${headline.trim()}</p>
                `;
                headlinesList.appendChild(li);
            }
        });

        if (headlinesList.innerHTML === "") {
            headlinesList.innerHTML = `<li class="text-center text-muted">No recent news headlines triggered this lookback window.</li>`;
        }

        // Fetch and render the technical indicator price chart
        renderModalPriceChart(row.ticker);

        // Open modal
        detailModal.classList.add("active");
    }

    async function renderModalPriceChart(ticker) {
        const ctx = document.getElementById("modal-price-chart").getContext("2d");
        
        if (modalPriceChart) {
            modalPriceChart.destroy();
            modalPriceChart = null;
        }

        try {
            const res = await fetch(`/api/ticker_history/${ticker}`);
            const history = await res.json();
            
            const labels = history.map(h => h.date);
            const closePrices = history.map(h => h.close);
            const sma20 = history.map(h => h.sma20);
            const sma50 = history.map(h => h.sma50);
            const sma200 = history.map(h => h.sma200);
            const support = history.map(h => h.support);
            const resistance = history.map(h => h.resistance);
            
            modalPriceChart = new Chart(ctx, {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: "Close Price",
                            data: closePrices,
                            borderColor: "#f3f4f6", // Light grey
                            borderWidth: 2,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: "SMA-20",
                            data: sma20,
                            borderColor: "hsl(42, 94%, 55%)", // Gold
                            borderWidth: 1.2,
                            pointRadius: 0,
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: "SMA-50",
                            data: sma50,
                            borderColor: "hsl(265, 85%, 65%)", // Purple
                            borderWidth: 1.2,
                            pointRadius: 0,
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: "SMA-200",
                            data: sma200,
                            borderColor: "hsl(200, 85%, 60%)", // Blue
                            borderWidth: 1.2,
                            pointRadius: 0,
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: "Support (20d)",
                            data: support,
                            borderColor: "hsl(346, 84%, 57%)", // Red dashed
                            borderWidth: 1,
                            borderDash: [4, 4],
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: "Resistance (20d)",
                            data: resistance,
                            borderColor: "hsl(162, 76%, 45%)", // Green dashed
                            borderWidth: 1,
                            borderDash: [4, 4],
                            pointRadius: 0,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: "top",
                            labels: {
                                color: "#9ca3af",
                                font: { family: "Inter", size: 9 },
                                boxWidth: 10
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: "rgba(255, 255, 255, 0.02)" },
                            ticks: { color: "#9ca3af", font: { family: "Inter", size: 8 }, maxTicksLimit: 8 }
                        },
                        y: {
                            grid: { color: "rgba(255, 255, 255, 0.02)" },
                            ticks: { color: "#9ca3af", font: { family: "Inter", size: 8 } }
                        }
                    }
                }
            });
            
        } catch (err) {
            console.error("Failed to render modal price chart:", err);
        }
    }

    function setScoreBar(scoreId, fillId, val) {
        const textElem = document.getElementById(scoreId);
        const fillElem = document.getElementById(fillId);
        
        textElem.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)}`;
        textElem.className = val > 0 ? "green-text" : (val < 0 ? "red-text" : "");
        
        // Map [-1, 1] to [0%, 100%]
        const pct = ((val + 1) / 2) * 100;
        fillElem.style.width = `${pct}%`;
        fillElem.style.backgroundColor = val > 0 ? "var(--color-buy)" : (val < 0 ? "var(--color-sell)" : "var(--text-muted)");
    }

    // ── Tab 5: Trader Tracker Logic ──────────────────────────
    const traderCallForm = document.getElementById("trader-call-form");
    if (traderCallForm) {
        traderCallForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                trader_name: document.getElementById("tc-trader-name").value.trim(),
                platform: document.getElementById("tc-platform").value,
                handle: document.getElementById("tc-handle").value.trim() || null,
                ticker: document.getElementById("tc-ticker").value.trim().toUpperCase(),
                call_type: document.getElementById("tc-call-type").value,
                price_at_call: parseFloat(document.getElementById("tc-price").value) || null,
                target_price: parseFloat(document.getElementById("tc-target").value) || null,
                timeframe: document.getElementById("tc-timeframe").value,
                notes: document.getElementById("tc-notes").value.trim() || null,
            };
            try {
                const res = await fetch("/api/traders/calls", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                traderCallForm.reset();
                loadTradersData();
            } catch (err) {
                console.error("Failed to log trader call:", err);
                alert("Failed to log call — check console for details.");
            }
        });
    }

    async function closeTraderCall(callId) {
        const exitPriceStr = prompt("Exit price for this call?");
        if (exitPriceStr === null) return;
        const exitPrice = parseFloat(exitPriceStr);
        if (Number.isNaN(exitPrice)) return;
        try {
            const res = await fetch(`/api/traders/calls/${callId}/close`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ exit_price: exitPrice }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            loadTradersData();
        } catch (err) {
            console.error("Failed to close trader call:", err);
        }
    }

    async function deleteTraderCall(callId) {
        if (!confirm("Delete this logged call?")) return;
        try {
            const res = await fetch(`/api/traders/calls/${callId}`, { method: "DELETE" });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            loadTradersData();
        } catch (err) {
            console.error("Failed to delete trader call:", err);
        }
    }
    window.closeTraderCall = closeTraderCall;
    window.deleteTraderCall = deleteTraderCall;

    async function loadTradersData() {
        try {
            const [summaryRes, callsRes] = await Promise.all([
                fetch("/api/traders/summary"),
                fetch("/api/traders/calls"),
            ]);
            const summary = await summaryRes.json();
            const calls = await callsRes.json();

            document.getElementById("traders-count-label").textContent = `${summary.length} trader${summary.length === 1 ? "" : "s"} tracked`;

            const summaryBody = document.getElementById("traders-summary-body");
            if (!summary.length) {
                summaryBody.innerHTML = `<tr><td colspan="7" class="text-center py-4">No traders tracked yet — log a call above.</td></tr>`;
            } else {
                summaryBody.innerHTML = "";
                summary.forEach(s => {
                    const tr = document.createElement("tr");
                    const wr = s.win_rate_pct;
                    const wrStr = wr === null || wr === undefined ? "—" : `${wr.toFixed(1)}%`;
                    const wrStyle = wr === null || wr === undefined ? "" : (wr >= 50 ? "green-text" : "red-text");
                    const avgStr = s.avg_outcome_pct === null || s.avg_outcome_pct === undefined ? "—" : `${s.avg_outcome_pct >= 0 ? "+" : ""}${s.avg_outcome_pct.toFixed(2)}%`;
                    tr.innerHTML = `
                        <td><strong>${s.trader_name}</strong></td>
                        <td><span class="badge badge-info">${s.platform}</span></td>
                        <td>${s.total_calls}</td>
                        <td>${s.open_calls}</td>
                        <td>${s.closed_calls}</td>
                        <td class="${wrStyle}">${wrStr}</td>
                        <td class="${wrStyle}">${avgStr}</td>
                    `;
                    summaryBody.appendChild(tr);
                });
            }

            const callsBody = document.getElementById("traders-calls-body");
            if (!calls.length) {
                callsBody.innerHTML = `<tr><td colspan="9" class="text-center py-4">No calls logged yet.</td></tr>`;
            } else {
                callsBody.innerHTML = "";
                calls.forEach(c => {
                    const tr = document.createElement("tr");
                    const outcomeStr = c.outcome_pct === null || c.outcome_pct === undefined ? "—" : `${c.outcome_pct >= 0 ? "+" : ""}${c.outcome_pct.toFixed(2)}%`;
                    const outcomeStyle = c.outcome_pct > 0 ? "green-text" : (c.outcome_pct < 0 ? "red-text" : "");
                    const typeClass = c.call_type === "SELL" ? "sell" : (c.call_type === "WATCH" ? "" : "buy");
                    const actions = c.status === "OPEN"
                        ? `<button class="action-link" onclick="closeTraderCall('${c.call_id}')">Close</button><button class="action-link danger" onclick="deleteTraderCall('${c.call_id}')">Delete</button>`
                        : `<button class="action-link danger" onclick="deleteTraderCall('${c.call_id}')">Delete</button>`;
                    tr.innerHTML = `
                        <td>${(c.called_at || "").slice(0, 10)}</td>
                        <td><strong>${c.trader_name}</strong><br><span style="color: var(--text-muted); font-size: 11px;">${c.platform}${c.handle ? " · " + c.handle : ""}</span></td>
                        <td>${c.ticker}</td>
                        <td><span class="rec-pill ${typeClass}" style="padding: 2px 8px;">${c.call_type}</span></td>
                        <td>${c.price_at_call ? "$" + c.price_at_call.toFixed(2) : "—"}</td>
                        <td>${c.target_price ? "$" + c.target_price.toFixed(2) : "—"}</td>
                        <td>${c.status}</td>
                        <td class="${outcomeStyle}">${outcomeStr}</td>
                        <td>${actions}</td>
                    `;
                    callsBody.appendChild(tr);
                });
            }
        } catch (err) {
            console.error("Traders tab load failed:", err);
        }
    }

    // ── Tab 6: Weekend Swing Signals Logic ───────────────────────
    let weekendScanCache = [];
    let weekendTriggeredCache = [];
    let weekendSortState = { field: "distance_pct", asc: true };
    let weekendSelectedSignalIds = new Set();
    let weekendPollHandle = null;

    const WEEKEND_STRATEGY_LABELS = {
        prior_week_breakout: "1: Prior-Week Breakout",
        overnight_range_breakout: "2: Overnight Range",
        gap_strength: "3: Gap Strength/Fade",
    };
    function weekendStrategyLabel(strategy) {
        return WEEKEND_STRATEGY_LABELS[strategy] || strategy;
    }
    function weekendPill(direction) {
        const cls = direction === "long" ? "buy" : "short";
        return `<span class="rec-pill ${cls}" style="padding: 2px 8px;">${direction.toUpperCase()}</span>`;
    }

    function startWeekendPolling() {
        stopWeekendPolling();
        weekendPollHandle = setInterval(() => {
            loadWeekendTriggered();
            loadWeekendActive();
        }, 20000);
    }
    function stopWeekendPolling() {
        if (weekendPollHandle) {
            clearInterval(weekendPollHandle);
            weekendPollHandle = null;
        }
    }

    // ── Market regime card (fear/greed + pressure + latest weekly review) ──
    function regimeEsc(v) {
        return String(v ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
    }

    async function loadRegime(runNow = false) {
        const body = document.getElementById("regime-body");
        if (!body) return;
        try {
            const res = await fetch(runNow ? "/api/regime/run" : "/api/regime", { method: runNow ? "POST" : "GET" });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "request failed");
            const r = data.latest;
            const reviewRes = await fetch("/api/weekend-swing/weekly-review");
            const review = (await reviewRes.json()).review;
            if (!r) {
                body.innerHTML = `No regime snapshot yet &mdash; click Refresh.`;
                return;
            }
            const fg = r.fear_greed == null ? "n/a" : Math.round(r.fear_greed);
            const dirColor = r.pressure_label === "BULLISH" ? "var(--color-buy)" : r.pressure_label === "BEARISH" ? "var(--color-sell)" : "var(--color-hold)";
            const sectors = (r.sectors || []);
            const fmtSec = s => `${regimeEsc(s.symbol)} ${s.flow >= 0 ? "+" : ""}${s.flow.toFixed(2)}`;
            body.innerHTML = `
                <div><strong>Fear/Greed ${fg}</strong> (${regimeEsc(r.fear_greed_label)}) &nbsp;|&nbsp;
                     <strong style="color:${dirColor}">Pressure ${r.pressure >= 0 ? "+" : ""}${r.pressure.toFixed(2)} ${regimeEsc(r.pressure_label)}</strong>
                     &nbsp;<span style="color:var(--text-muted)">as of ${regimeEsc(r.as_of)}</span></div>
                <ul style="margin: 8px 0 0 18px;">${(r.summary || []).map(l => `<li>${regimeEsc(l)}</li>`).join("")}</ul>
                ${sectors.length ? `<div style="margin-top:6px; color:var(--text-muted);">Sector flow: buying ${sectors.slice(0, 3).map(fmtSec).join(", ")} | selling ${sectors.slice(-3).reverse().map(fmtSec).join(", ")}</div>` : ""}
                ${review ? `<div style="margin-top:10px;"><strong>Weekly review (${regimeEsc(review.week_start)})</strong><ul style="margin: 4px 0 0 18px;">${(review.findings || []).map(f => `<li>${regimeEsc(f)}</li>`).join("")}</ul></div>` : ""}`;
        } catch (err) {
            body.textContent = `Could not load regime: ${err.message}`;
        }
    }

    document.getElementById("btn-regime-refresh")?.addEventListener("click", () => loadRegime(true));

    async function loadWeekendTabData() {
        await Promise.all([loadWeekendScan(), loadWeekendTriggered(), loadWeekendActive(), loadRegime()]);
    }

    // -- Scan Results --
    async function loadWeekendScan() {
        try {
            const res = await fetch("/api/weekend-swing/scan");
            const data = await res.json();
            weekendScanCache = data.results || [];
            renderWeekendScanTable();
        } catch (err) {
            console.error("Weekend scan load failed:", err);
        }
    }

    function renderWeekendScanTable() {
        const body = document.getElementById("weekend-scan-body");
        document.getElementById("weekend-scan-count-label").textContent = `${weekendScanCache.length} candidates loaded`;

        if (!weekendScanCache.length) {
            body.innerHTML = `<tr><td colspan="8" class="text-center py-4">No scan loaded yet &mdash; click "Run Weekend Scan".</td></tr>`;
            return;
        }

        const field = weekendSortState.field;
        const dir = weekendSortState.asc ? 1 : -1;
        const sorted = [...weekendScanCache].sort((a, b) => {
            if (field === "ticker") return a.ticker.localeCompare(b.ticker) * dir;
            return ((a[field] || 0) - (b[field] || 0)) * dir;
        });

        body.innerHTML = "";
        sorted.forEach(c => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${c.ticker}</strong></td>
                <td>${weekendStrategyLabel(c.strategy)}</td>
                <td>$${(c.pivot_high || 0).toFixed(2)}</td>
                <td>$${(c.pivot_low || 0).toFixed(2)}</td>
                <td>$${(c.current_price || 0).toFixed(2)}</td>
                <td>${(c.distance_pct || 0).toFixed(2)}%</td>
                <td>${c.earnings_excluded ? '<span class="badge" style="background: rgba(239,68,68,0.15); color: var(--color-sell);">EARNINGS</span>' : ''}</td>
                <td class="text-muted"><small>${c.data_quality || ''}</small></td>
            `;
            body.appendChild(tr);
        });
    }

    document.querySelectorAll('#weekend-scan-table .sortable').forEach(th => {
        th.addEventListener("click", () => {
            const sortField = th.getAttribute("data-sort");
            if (weekendSortState.field === sortField) {
                weekendSortState.asc = !weekendSortState.asc;
            } else {
                weekendSortState = { field: sortField, asc: true };
            }
            renderWeekendScanTable();
        });
    });

    const btnWeekendRunScan = document.getElementById("btn-weekend-run-scan");
    if (btnWeekendRunScan) {
        btnWeekendRunScan.addEventListener("click", async () => {
            btnWeekendRunScan.disabled = true;
            btnWeekendRunScan.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scanning...`;
            try {
                const res = await fetch("/api/weekend-swing/scan/run", { method: "POST" });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                await loadWeekendScan();
            } catch (err) {
                console.error("Weekend scan run failed:", err);
                alert(`Weekend scan failed: ${err.message}`);
            } finally {
                btnWeekendRunScan.disabled = false;
                btnWeekendRunScan.innerHTML = `<i class="fa-solid fa-play"></i> Run Weekend Scan`;
            }
        });
    }

    // -- Monday Trigger Feed --
    async function loadWeekendTriggered() {
        try {
            const res = await fetch("/api/weekend-swing/triggered");
            const data = await res.json();
            weekendTriggeredCache = data.signals || [];
            const liveIds = new Set(weekendTriggeredCache.map(s => s.signal_id));
            weekendSelectedSignalIds.forEach(id => { if (!liveIds.has(id)) weekendSelectedSignalIds.delete(id); });
            renderWeekendTriggeredTable();
        } catch (err) {
            console.error("Weekend triggered load failed:", err);
        }
    }

    function renderWeekendTriggeredTable() {
        const body = document.getElementById("weekend-triggered-body");
        document.getElementById("weekend-triggered-count-label").textContent = `${weekendTriggeredCache.length} triggered`;

        if (!weekendTriggeredCache.length) {
            body.innerHTML = `<tr><td colspan="7" class="text-center py-4">No signals triggered yet &mdash; click "Check Triggers Now".</td></tr>`;
        } else {
            body.innerHTML = "";
            weekendTriggeredCache.forEach(s => {
                const gapNote = s.gap_type ? ` <span class="text-muted" style="font-size:10px;">(${s.gap_type.replace(/_/g, "-")})</span>` : "";
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><input type="checkbox" class="weekend-signal-checkbox" data-signal-id="${s.signal_id}" ${weekendSelectedSignalIds.has(s.signal_id) ? "checked" : ""}></td>
                    <td><strong>${s.ticker}</strong></td>
                    <td>${weekendStrategyLabel(s.strategy)}${gapNote}</td>
                    <td>${weekendPill(s.direction)}</td>
                    <td>$${(s.trigger_price || 0).toFixed(2)}</td>
                    <td>$${(s.stop_price || 0).toFixed(2)}</td>
                    <td class="text-muted"><small>${(s.triggered_at || "").replace("T", " ").slice(0, 16)}</small></td>
                `;
                body.appendChild(tr);
            });

            document.querySelectorAll(".weekend-signal-checkbox").forEach(cb => {
                cb.addEventListener("change", (e) => {
                    const id = e.target.getAttribute("data-signal-id");
                    if (e.target.checked) weekendSelectedSignalIds.add(id);
                    else weekendSelectedSignalIds.delete(id);
                    updateWeekendSubmitButton();
                });
            });
        }
        updateWeekendSubmitButton();
    }

    function updateWeekendSubmitButton() {
        const btn = document.getElementById("btn-weekend-submit-selected");
        if (!btn) return;
        btn.disabled = weekendSelectedSignalIds.size === 0;
        btn.innerHTML = weekendSelectedSignalIds.size
            ? `<i class="fa-solid fa-paper-plane"></i> Submit ${weekendSelectedSignalIds.size} Selected`
            : `<i class="fa-solid fa-paper-plane"></i> Submit Selected to Paper Trading`;
    }

    document.getElementById("weekend-select-all")?.addEventListener("change", (e) => {
        weekendTriggeredCache.forEach(s => {
            if (e.target.checked) weekendSelectedSignalIds.add(s.signal_id);
            else weekendSelectedSignalIds.delete(s.signal_id);
        });
        renderWeekendTriggeredTable();
    });

    const btnWeekendCheckTriggers = document.getElementById("btn-weekend-check-triggers");
    if (btnWeekendCheckTriggers) {
        btnWeekendCheckTriggers.addEventListener("click", async () => {
            btnWeekendCheckTriggers.disabled = true;
            btnWeekendCheckTriggers.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Checking...`;
            try {
                const res = await fetch("/api/weekend-swing/triggered/check", { method: "POST" });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                await loadWeekendTriggered();
            } catch (err) {
                console.error("Trigger check failed:", err);
                alert(`Trigger check failed: ${err.message}`);
            } finally {
                btnWeekendCheckTriggers.disabled = false;
                btnWeekendCheckTriggers.innerHTML = `<i class="fa-solid fa-bolt"></i> Check Triggers Now`;
            }
        });
    }

    // -- Submit-to-paper-trade flow (explicit confirm, no one-click submits) --
    const weekendSubmitModal = document.getElementById("weekend-submit-modal");
    document.getElementById("btn-weekend-submit-selected")?.addEventListener("click", () => {
        if (!weekendSelectedSignalIds.size) return;
        const body = document.getElementById("weekend-submit-confirm-body");
        body.innerHTML = "";
        weekendTriggeredCache.filter(s => weekendSelectedSignalIds.has(s.signal_id)).forEach(s => {
            const sizeEst = s.direction === "long"
                ? `~$1,000 notional`
                : `~${Math.floor(1000 / s.trigger_price)} sh (~$1,000)`;
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${s.ticker}</strong></td>
                <td>${weekendStrategyLabel(s.strategy)}</td>
                <td>${weekendPill(s.direction)}</td>
                <td>$${s.trigger_price.toFixed(2)}</td>
                <td>$${s.stop_price.toFixed(2)}</td>
                <td>${sizeEst}</td>
            `;
            body.appendChild(tr);
        });
        weekendSubmitModal.classList.add("active");
    });

    function closeWeekendSubmitModal() {
        weekendSubmitModal.classList.remove("active");
    }
    document.getElementById("btn-close-weekend-submit-modal")?.addEventListener("click", closeWeekendSubmitModal);
    document.getElementById("btn-weekend-submit-cancel")?.addEventListener("click", closeWeekendSubmitModal);

    document.getElementById("btn-weekend-submit-confirm")?.addEventListener("click", async () => {
        const btn = document.getElementById("btn-weekend-submit-confirm");
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Submitting...`;
        try {
            const res = await fetch("/api/weekend-swing/submit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ signal_ids: Array.from(weekendSelectedSignalIds) }),
            });
            const data = await res.json();
            if (data.failed && data.failed.length) {
                alert(`Some signals failed to submit:\n${data.failed.map(f => `${f.signal_id.slice(0, 8)}: ${f.reason}`).join("\n")}`);
            }
            weekendSelectedSignalIds.clear();
            closeWeekendSubmitModal();
            await Promise.all([loadWeekendTriggered(), loadWeekendActive()]);
        } catch (err) {
            console.error("Submit failed:", err);
            alert("Submission failed — check console / server logs.");
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-check"></i> Confirm & Submit`;
        }
    });

    // -- Active Positions --
    async function loadWeekendActive() {
        try {
            const res = await fetch("/api/weekend-swing/active");
            const data = await res.json();
            const positions = data.positions || [];
            const body = document.getElementById("weekend-active-body");
            document.getElementById("weekend-active-count-label").textContent = `${positions.length} open`;

            if (!positions.length) {
                body.innerHTML = `<tr><td colspan="8" class="text-center py-4">No active positions.</td></tr>`;
                return;
            }
            body.innerHTML = "";
            positions.forEach(p => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${p.ticker}</strong></td>
                    <td>${weekendStrategyLabel(p.strategy)}</td>
                    <td>${weekendPill(p.direction)}</td>
                    <td>$${(p.entry_price || 0).toFixed(2)}</td>
                    <td>${(p.qty || 0).toFixed(4).replace(/\.?0+$/, "")}</td>
                    <td>$${(p.notional || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td class="text-muted"><small>${(p.submitted_at || "").replace("T", " ").slice(0, 16)}</small></td>
                    <td><button class="action-link danger" data-signal-id="${p.signal_id}">Close</button></td>
                `;
                body.appendChild(tr);
            });

            document.querySelectorAll("#weekend-active-body .action-link").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    const signalId = e.target.getAttribute("data-signal-id");
                    if (!confirm("Close this position at the current market price?")) return;
                    try {
                        const res = await fetch(`/api/weekend-swing/close/${signalId}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ exit_reason: "manual" }),
                        });
                        if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || `HTTP ${res.status}`);
                        }
                        await loadWeekendActive();
                    } catch (err) {
                        console.error("Close failed:", err);
                        alert(`Close failed: ${err.message}`);
                    }
                });
            });
        } catch (err) {
            console.error("Weekend active load failed:", err);
        }
    }

    // -- History modal --
    const weekendHistoryModal = document.getElementById("weekend-history-modal");
    document.getElementById("btn-weekend-history")?.addEventListener("click", () => {
        loadWeekendHistory();
        weekendHistoryModal.classList.add("active");
    });
    document.getElementById("btn-close-weekend-history-modal")?.addEventListener("click", () => {
        weekendHistoryModal.classList.remove("active");
    });
    document.getElementById("btn-weekend-history-apply-filters")?.addEventListener("click", () => loadWeekendHistory());

    async function loadWeekendHistory() {
        const strategy = document.getElementById("weekend-history-strategy-filter").value;
        const dateFrom = document.getElementById("weekend-history-date-from").value;
        const dateTo = document.getElementById("weekend-history-date-to").value;

        const params = new URLSearchParams();
        if (strategy) params.set("strategy", strategy);
        if (dateFrom) params.set("date_from", dateFrom);
        if (dateTo) params.set("date_to", dateTo);

        try {
            const res = await fetch(`/api/weekend-swing/history?${params.toString()}`);
            const data = await res.json();
            renderWeekendHistorySummary(data.summary);
            renderWeekendHistoryTable(data.trades || []);
        } catch (err) {
            console.error("Weekend history load failed:", err);
        }
    }

    function renderWeekendHistorySummary(summary) {
        const container = document.getElementById("weekend-history-summary");
        const c = summary.combined;
        const wr = c.win_rate_pct;
        const wrColor = wr === null ? "" : (wr >= 50 ? "var(--color-buy)" : "var(--color-sell)");
        container.innerHTML = `
            <div class="glass-card metric-card">
                <div class="metric-icon"><i class="fa-solid fa-list"></i></div>
                <div class="metric-data"><span class="metric-title">Total Trades</span><h2>${c.total_trades}</h2></div>
            </div>
            <div class="glass-card metric-card">
                <div class="metric-icon" style="color:${wrColor};"><i class="fa-solid fa-trophy"></i></div>
                <div class="metric-data"><span class="metric-title">Win Rate</span><h2 style="color:${wrColor};">${wr === null ? "--" : wr.toFixed(1) + "%"}</h2></div>
            </div>
            <div class="glass-card metric-card">
                <div class="metric-icon positive"><i class="fa-solid fa-arrow-trend-up"></i></div>
                <div class="metric-data"><span class="metric-title">Avg Win</span><h2 class="green-text">${c.avg_win_pct === null ? "--" : "+" + c.avg_win_pct.toFixed(2) + "%"}</h2></div>
            </div>
            <div class="glass-card metric-card">
                <div class="metric-icon negative"><i class="fa-solid fa-arrow-trend-down"></i></div>
                <div class="metric-data"><span class="metric-title">Avg Loss</span><h2 class="red-text">${c.avg_loss_pct === null ? "--" : c.avg_loss_pct.toFixed(2) + "%"}</h2></div>
            </div>
        `;
    }

    function renderWeekendHistoryTable(trades) {
        const body = document.getElementById("weekend-history-body");
        if (!trades.length) {
            body.innerHTML = `<tr><td colspan="8" class="text-center py-4">No closed trades match these filters.</td></tr>`;
            return;
        }
        body.innerHTML = "";
        trades.forEach(t => {
            const pl = t.outcome_pct || 0;
            const plClass = pl > 0 ? "green-text" : "red-text";
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="text-muted"><small>${(t.triggered_at || "").slice(0, 10)}</small></td>
                <td><strong>${t.ticker}</strong></td>
                <td>${weekendStrategyLabel(t.strategy)}</td>
                <td>${weekendPill(t.direction)}</td>
                <td>$${(t.entry_price || 0).toFixed(2)}</td>
                <td>$${(t.exit_price || 0).toFixed(2)}</td>
                <td class="text-muted"><small>${t.exit_reason || ""}</small></td>
                <td class="${plClass}">${pl >= 0 ? "+" : ""}${pl.toFixed(2)}%</td>
            `;
            body.appendChild(tr);
        });
    }
});
