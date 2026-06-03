"use strict";
// Global Variables
let map;
let markersGroup;
let currentTab = 'reputation';
let allRestaurants = [];
// Initialize Dashboard on Load
window.addEventListener('DOMContentLoaded', () => {
    initMap();
    fetchRestaurants();
    checkExistingEvaluation();
    initSearch();
});
// 1. Map Initialization
function initMap() {
    // Center map around Pathum Wan / Phra Nakhon area in Bangkok
    map = L.map('map', {
        zoomControl: false
    }).setView([13.7380, 100.5350], 13);
    // Zoom buttons position
    L.control.zoom({
        position: 'bottomleft'
    }).addTo(map);
    // Premium Vibrant CartoDB Voyager tiles (Highly Colorful)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);
    markersGroup = L.layerGroup().addTo(map);
}
// 2. Fetch and Render Restaurant Pins
async function fetchRestaurants() {
    try {
        const response = await fetch('/api/restaurants');
        if (!response.ok)
            throw new Error('Failed to load restaurants');
        const restaurants = await response.json();
        allRestaurants = restaurants;
        plotPins(restaurants);
        const tracked = document.getElementById('stat-tracked');
        const anomalies = document.getElementById('stat-anomalies');
        if (tracked) tracked.textContent = restaurants.length;
        if (anomalies) anomalies.textContent = restaurants.filter(r => r.has_anomaly).length;
    }
    catch (err) {
        console.error('Error fetching restaurants:', err);
    }
}
function plotPins(restaurants) {
    markersGroup.clearLayers();
    restaurants.forEach((r) => {
        // Create custom neon circular pin HTML
        const customPinHTML = `
            <div class="custom-map-pin ${r.has_anomaly ? 'anomaly-pin' : ''}" style="
                background-color: ${r.ai_hex_color};
                box-shadow: 0 0 14px ${r.ai_hex_color}99, 0 2px 6px rgba(0,0,0,0.25);
            ">
                <span class="pin-score">${r.avg_ai_rating.toFixed(1)}</span>
            </div>
        `;
        const pinIcon = L.divIcon({
            html: customPinHTML,
            className: 'custom-pin-container',
            iconSize: [44, 44],
            iconAnchor: [22, 22]
        });
        const marker = L.marker([r.lat, r.lng], { icon: pinIcon });
        // Pin interaction events
        marker.on('click', () => {
            selectRestaurant(r.id);
            map.panTo([r.lat, r.lng]);
        });
        // Simple Leaflet popup on hover
        marker.bindTooltip(`
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.82rem; padding: 0.25rem 0.5rem; background: #060913; color: #f8fafc; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;">
                <strong>${r.name}</strong><br>
                <span style="color: #94a3b8; font-size: 0.72rem;">User: ${r.avg_user_rating}★ | AI: ${r.avg_ai_rating}★</span>
            </div>
        `, {
            direction: 'top',
            offset: [0, -10],
            opacity: 0.95
        });
        markersGroup.addLayer(marker);
    });
}
// 3. Select and Load Restaurant Intelligence Details
async function selectRestaurant(id) {
    // Show details section and hide empty placeholder state
    const placeholder = document.getElementById('reputation-placeholder');
    if (placeholder)
        placeholder.classList.add('hidden');
    const detailsContainer = document.getElementById('reputation-details');
    if (detailsContainer)
        detailsContainer.classList.remove('hidden');
    try {
        const response = await fetch(`/api/restaurants/${id}`);
        if (!response.ok)
            throw new Error('Failed to fetch restaurant details');
        const r = await response.json();
        // Populate text meta
        const restName = document.getElementById('rest-name');
        const restCuisine = document.getElementById('rest-cuisine');
        const restAddress = document.getElementById('rest-address');
        const restDesc = document.getElementById('rest-desc');
        const userRatingVal = document.getElementById('user-rating-val');
        const aiRatingVal = document.getElementById('ai-rating-val');
        const userRatingStars = document.getElementById('user-rating-stars');
        const aiRatingStars = document.getElementById('ai-rating-stars');
        const totalReviewsCount = document.getElementById('total-reviews-count');
        const anomaliesCount = document.getElementById('anomalies-count');
        if (restName)
            restName.innerText = r.name;
        if (restCuisine)
            restCuisine.innerText = r.cuisine;
        if (restAddress)
            restAddress.innerHTML = `<i class="fa-solid fa-location-dot"></i> ${r.address}`;
        if (restDesc)
            restDesc.innerText = r.description;
        // Ratings stats
        if (userRatingVal)
            userRatingVal.innerText = r.avg_user_rating.toFixed(2);
        if (aiRatingVal)
            aiRatingVal.innerText = r.avg_ai_rating.toFixed(2);
        // Stars rendering
        if (userRatingStars)
            userRatingStars.innerHTML = getStarRatingHTML(r.avg_user_rating);
        if (aiRatingStars)
            aiRatingStars.innerHTML = getStarRatingHTML(r.avg_ai_rating);
        // Counters
        if (totalReviewsCount)
            totalReviewsCount.innerText = r.review_count.toString();
        if (anomaliesCount)
            anomaliesCount.innerText = r.anomaly_count.toString();
        // Anomaly warnings
        const fraudBanner = document.getElementById('fraud-alert-banner');
        if (fraudBanner) {
            if (r.has_anomaly) {
                fraudBanner.classList.remove('hidden');
            }
            else {
                fraudBanner.classList.add('hidden');
            }
        }
        // Render detailed reviews list logs
        if (r.reviews) {
            renderReviewLogs(r.reviews);
        }
    }
    catch (err) {
        console.error('Error loading details:', err);
    }
}
// Helper to draw stars dynamically based on ratings (out of 5 stars)
function getStarRatingHTML(rating) {
    let stars = '';
    const rounded = Math.round(rating * 2) / 2; // Round to nearest 0.5
    for (let i = 1; i <= 5; i++) {
        if (i <= rounded) {
            stars += '<i class="fa-solid fa-star"></i>';
        }
        else if (i - 0.5 === rounded) {
            stars += '<i class="fa-solid fa-star-half-stroke"></i>';
        }
        else {
            stars += '<i class="fa-regular fa-star"></i>';
        }
    }
    return stars;
}
// Render Review Log List Cards
function renderReviewLogs(reviews) {
    const listContainer = document.getElementById('reviews-list-container');
    if (!listContainer)
        return;
    listContainer.innerHTML = ''; // Clear logs
    reviews.forEach((rev) => {
        const isAnomaly = rev.is_anomaly;
        const deltaFormatted = rev.delta.toFixed(2);
        const aiExpectedFormatted = rev.ai_expected_rating.toFixed(2);
        const card = document.createElement('div');
        card.className = `review-card ${isAnomaly ? 'anomaly-card' : ''}`;
        card.innerHTML = `
            <div class="review-header-meta">
                <div class="stars-row">
                    <span class="stars-lbl"><i class="fa-regular fa-user"></i> User: ${getStarRatingHTML(rev.user_rating)}</span>
                </div>
                ${isAnomaly ? `
                    <span class="anomaly-tag">
                        <i class="fa-solid fa-circle-exclamation"></i> Suspicious
                    </span>
                ` : ''}
            </div>
            
            <p class="review-text">${rev.text}</p>
            
            <div class="review-footer-metrics">
                <span class="metric-inline">Google User: <span class="val-highlight user-highlight">${rev.user_rating}★</span></span>
                <span class="metric-inline">AI Expected: <span class="val-highlight ai-highlight">${aiExpectedFormatted}★</span></span>
                <span class="metric-inline">Delta (&Delta;): <span class="val-highlight delta-highlight">${deltaFormatted}</span></span>
            </div>
        `;
        listContainer.appendChild(card);
    });
}
// 4. Navigation Tab Controllers
function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    const tabBtn = document.getElementById(`tab-btn-${tabName}`);
    if (tabBtn) tabBtn.classList.add('active');
    const intelligenceView = document.getElementById('intelligence-view');
    const evaluationView = document.getElementById('evaluation-view');
    if (tabName === 'reputation') {
        intelligenceView?.classList.remove('hidden');
        evaluationView?.classList.add('hidden');
        setTimeout(() => map.invalidateSize(), 50);
    } else {
        intelligenceView?.classList.add('hidden');
        evaluationView?.classList.remove('hidden');
    }
}
// 5. Evaluation Dashboard
const MODEL_META = {
    xlmr:      { name: 'XLM-R',       sub: 'xlm-roberta · fine-tuned',   color: '#0d9488' },
    embedding: { name: 'Embedding',   sub: 'E5/BGE + classifier head',   color: '#7c3aed' },
    baseline:  { name: 'Baseline',    sub: 'TF-IDF + XGBoost',           color: '#64748b' },
};
let evalFocusKey = null;
let evalModelsData = null;
async function checkExistingEvaluation() {
    try {
        const response = await fetch('/api/evaluation-results');
        if (response.ok) {
            const results = await response.json();
            displayEvaluationResults(results);
        }
    }
    catch (err) { /* no eval file yet */ }
}
function displayEvaluationResults(results) {
    const emptyState = document.getElementById('evaluation-empty-state');
    const resultsDash = document.getElementById('evaluation-results-dashboard');
    if (emptyState) emptyState.style.display = 'none';
    if (resultsDash) resultsDash.style.display = 'flex';
    const rawModels = results.models || {};
    const bestKey = results.best_model || null;
    const sorted = Object.entries(rawModels)
        .map(([key, data]) => ({ key, data }))
        .sort((a, b) => b.data.accuracy - a.data.accuracy);
    evalModelsData = rawModels;
    evalFocusKey = bestKey || sorted[0]?.key || null;
    renderEvalModelList(sorted, bestKey);
    const champion = sorted.find(m => m.key === bestKey) || sorted[0];
    renderEvalChampion(champion);
    renderEvalHeatTable(sorted);
    if (evalFocusKey) renderEvalFocus(evalFocusKey);
    const activeLabel = document.getElementById('active-model-label');
    if (activeLabel && bestKey) {
        const meta = MODEL_META[bestKey] || { name: bestKey, sub: '' };
        activeLabel.textContent = `${meta.name} · ${meta.sub}`;
    }
}
function renderEvalModelList(sorted, bestKey) {
    const container = document.getElementById('eval-model-list');
    if (!container) return;
    container.innerHTML = sorted.map(({ key, data }) => {
        const meta = MODEL_META[key] || { name: key, sub: '', color: '#94a3b8' };
        const isBest = key === bestKey;
        const accPct = (data.accuracy * 100).toFixed(1);
        const barW = (data.accuracy * 100).toFixed(1);
        const isActive = key === evalFocusKey;
        return `<div style="padding:10px 11px;border-radius:10px;border:${isActive ? `2px solid ${meta.color}` : '1px solid #e2e8f0'};background:${isActive ? meta.color + '12' : '#fff'};cursor:pointer;transition:all .15s" data-model-key="${key}" onclick="selectEvalModel('${key}')">
            <div style="display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center">
                <span style="width:10px;height:10px;border-radius:3px;background:${meta.color};display:inline-block;flex-shrink:0"></span>
                <div style="min-width:0">
                    <div style="font-size:12.5px;font-weight:600;color:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${meta.name}${isBest ? ' 🏆' : ''}</div>
                    <div style="font-size:10px;color:#94a3b8;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${meta.sub}</div>
                </div>
                <span style="font-family:monospace;font-size:11px;font-weight:700;color:#475569">${accPct}%</span>
            </div>
            <div style="margin-top:8px;height:5px;border-radius:99px;background:#f1f5f9;overflow:hidden">
                <div style="height:100%;width:${barW}%;background:${meta.color};border-radius:99px"></div>
            </div>
        </div>`;
    }).join('');
}
function selectEvalModel(key) {
    evalFocusKey = key;
    if (evalModelsData) {
        const sorted = Object.entries(evalModelsData).map(([k, d]) => ({ key: k, data: d })).sort((a, b) => b.data.accuracy - a.data.accuracy);
        renderEvalModelList(sorted, null);
    }
    renderEvalFocus(key);
}
window.selectEvalModel = selectEvalModel;
function renderEvalChampion(champion) {
    const el = document.getElementById('eval-champion');
    if (!el || !champion) return;
    const meta = MODEL_META[champion.key] || { name: champion.key, sub: '', color: '#f97316' };
    const d = champion.data;
    el.style.background = `linear-gradient(150deg,${meta.color}14,white 72%)`;
    el.style.borderColor = `${meta.color}44`;
    el.innerHTML = `<div style="flex:1"><span style="display:inline-block;font-family:monospace;font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:4px 10px;border-radius:99px;border:1px solid ${meta.color}44;color:${meta.color};background:white;margin-bottom:8px">🏆 Best Overall · Active</span>
        <div style="font-family:'Unbounded',sans-serif;font-size:19px;font-weight:700;color:#0f172a;line-height:1.1">${meta.name}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:3px">${meta.sub}</div></div>
    <div style="display:flex;gap:28px;flex-shrink:0;text-align:right">
        <div><div style="font-family:monospace;font-size:21px;font-weight:600;color:${meta.color};line-height:1">${(d.accuracy * 100).toFixed(1)}%</div><div style="font-family:monospace;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-top:4px">Accuracy</div></div>
        <div><div style="font-family:monospace;font-size:21px;font-weight:600;color:${meta.color};line-height:1">${d.f1_macro.toFixed(3)}</div><div style="font-family:monospace;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-top:4px">F1 Macro</div></div>
        <div><div style="font-family:monospace;font-size:21px;font-weight:600;color:${meta.color};line-height:1">${d.mae.toFixed(3)}</div><div style="font-family:monospace;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-top:4px">MAE ↓</div></div>
        <div><div style="font-family:monospace;font-size:21px;font-weight:600;color:${meta.color};line-height:1">${d.rmse.toFixed(3)}</div><div style="font-family:monospace;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-top:4px">RMSE ↓</div></div>
    </div>`;
}
function renderEvalHeatTable(sorted) {
    const table = document.getElementById('eval-heat-table');
    if (!table) return;
    const cols = [
        { key: 'mae',         label: 'MAE↓',  lower: true,  fmt: 4 },
        { key: 'rmse',        label: 'RMSE↓', lower: true,  fmt: 4 },
        { key: 'accuracy',    label: 'Acc↑',  lower: false, fmt: 3 },
        { key: 'f1_macro',    label: 'F1m↑',  lower: false, fmt: 3 },
        { key: 'f1_weighted', label: 'F1w↑',  lower: false, fmt: 3 },
    ];
    const colRanges = cols.map(c => {
        const vals = sorted.map(m => m.data[c.key] ?? 0);
        return { min: Math.min(...vals), max: Math.max(...vals) };
    });
    const starRanges = [1,2,3,4,5].map(s => {
        const vals = sorted.map(m => m.data.classification_report?.[s.toString()]?.['f1-score'] ?? 0);
        return { min: Math.min(...vals), max: Math.max(...vals) };
    });
    function computeF(val, range, lower) {
        let f = (val - range.min) / ((range.max - range.min) || 1);
        if (lower) f = 1 - f;
        return f;
    }
    // หา index ของแถวที่ชนะในแต่ละคอลัมน์ (ตัวเดียวต่อคอลัมน์)
    function bestRowIdx(vals, lower) {
        let bi = 0;
        for (let i = 1; i < vals.length; i++) {
            if (lower ? vals[i] < vals[bi] : vals[i] > vals[bi]) bi = i;
        }
        return bi;
    }
    const colWinners = cols.map(c => bestRowIdx(sorted.map(m => m.data[c.key] ?? 0), c.lower));
    const starWinners = [1,2,3,4,5].map(s => bestRowIdx(sorted.map(m => m.data.classification_report?.[s.toString()]?.['f1-score'] ?? 0), false));
    function heatCell(val, range, lower, fmt, borderLeft = false, isWinner = false) {
        const f = computeF(val, range, lower);
        const hue = lower ? 200 : 256;
        const bg = f < 0.05 ? '#f8fafc' : `oklch(${(0.975 - f * 0.45).toFixed(3)} ${(0.012 + f * 0.14).toFixed(3)} ${hue})`;
        const color = f > 0.45 ? '#fff' : '#334155';
        const outline = isWinner ? 'box-shadow:inset 0 0 0 2.5px #0f172a;' : '';
        const bl = borderLeft ? 'border-left:1px solid #e2e8f0;' : '';
        return `<td style="${bl}padding:0;text-align:center"><div style="display:grid;place-items:center;height:28px;border-radius:6px;background:${bg};color:${color};font-family:monospace;font-size:11px;font-weight:600;${outline}">${val.toFixed(fmt)}</div></td>`;
    }
    const thStyle = 'font-family:monospace;font-size:9px;color:#94a3b8;padding:4px 4px 8px;font-weight:500;text-align:center';
    const thead = `<thead><tr>
        <th style="${thStyle};text-align:left;padding-left:6px">Model</th>
        ${cols.map(c => `<th style="${thStyle}">${c.label}</th>`).join('')}
        ${[1,2,3,4,5].map((s, i) => `<th style="${thStyle}${i === 0 ? ';border-left:1px solid #e2e8f0' : ''}">${s}★</th>`).join('')}
    </tr></thead>`;
    const tbody = `<tbody>${sorted.map(({ key, data }, rowIdx) => {
        const meta = MODEL_META[key] || { name: key, color: '#94a3b8' };
        const cr = data.classification_report || {};
        const colCells = cols.map((c, i) =>
            heatCell(data[c.key] ?? 0, colRanges[i], c.lower, c.fmt, false, colWinners[i] === rowIdx)
        ).join('');
        const starCells = [1,2,3,4,5].map((s, i) =>
            heatCell(cr[s.toString()]?.['f1-score'] ?? 0, starRanges[i], false, 2, i === 0, starWinners[i] === rowIdx)
        ).join('');
        const isActive = key === evalFocusKey;
        return `<tr onclick="selectEvalModel('${key}')" style="cursor:pointer;transition:background .12s${isActive ? ';background:#f0f9ff' : ''}" onmouseenter="this.style.background='#f8fafc'" onmouseleave="this.style.background='${isActive ? '#f0f9ff' : ''}'">
            <td style="padding:0;padding-right:10px;white-space:nowrap">
                <div style="display:flex;align-items:center;gap:7px">
                    <span style="width:9px;height:9px;border-radius:3px;background:${meta.color};display:inline-block;flex-shrink:0"></span>
                    <span style="font-size:12px;font-weight:600;color:#1e293b">${meta.name}</span>
                </div>
            </td>${colCells}${starCells}</tr>`;
    }).join('')}</tbody>`;
    table.innerHTML = thead + tbody;
}
function renderEvalFocus(key) {
    const data = evalModelsData?.[key];
    if (!data) return;
    const meta = MODEL_META[key] || { name: key, sub: '', color: '#94a3b8' };
    const header = document.getElementById('eval-focus-header');
    if (header) {
        header.innerHTML = `
            <span style="width:13px;height:13px;border-radius:4px;background:${meta.color};display:inline-block;flex-shrink:0"></span>
            <div><div style="font-size:15px;font-weight:700;color:#0f172a">${meta.name}</div><div style="font-size:11px;color:#94a3b8;margin-top:1px">${meta.sub}</div></div>
            <span style="margin-left:auto;font-family:monospace;font-size:10.5px;color:#475569;background:#f8fafc;border:1px solid #e2e8f0;border-radius:99px;padding:5px 12px;white-space:nowrap">
                Acc ${(data.accuracy * 100).toFixed(1)}% · F1 ${data.f1_macro.toFixed(3)} · MAE ${data.mae.toFixed(3)} · RMSE ${data.rmse.toFixed(3)}
            </span>`;
    }
    const body = document.getElementById('eval-focus-body');
    if (!body) return;
    const cm = data.confusion_matrix || [];
    const starLabels = ['1★','2★','3★','4★','5★'];
    let cmHtml = '';
    if (cm.length > 0) {
        const maxVal = Math.max(...cm.flat(), 1);
        cmHtml = `<div>
            <div style="font-family:monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:10px">Confusion Matrix · Actual × Predicted</div>
            <table style="border-collapse:separate;border-spacing:2px">
                <thead><tr>
                    <th style="font-size:9px;color:#94a3b8;padding:0 8px 6px 0;text-align:left">Actual╲Predicted</th>
                    ${starLabels.map(s => `<th style="font-size:9px;color:#94a3b8;padding:0 2px 6px;text-align:center;width:36px">${s}</th>`).join('')}
                </tr></thead>
                <tbody>${cm.map((row, ri) => `<tr>
                    <td style="font-size:10px;color:#94a3b8;font-family:monospace;padding-right:8px;text-align:right">${starLabels[ri]}</td>
                    ${row.map((v, ci) => {
                        const t = v / maxVal;
                        const isDiag = ri === ci;
                        const bg = isDiag ? `rgba(2,132,199,${0.08 + t * 0.72})` : `rgba(15,23,42,${t * 0.08})`;
                        const color = isDiag && t > 0.45 ? '#fff' : isDiag ? '#0284c7' : '#334155';
                        return `<td style="padding:0;text-align:center"><div style="width:36px;height:26px;display:grid;place-items:center;border-radius:5px;background:${bg};color:${color};font-family:monospace;font-size:11px;font-weight:600">${v}</div></td>`;
                    }).join('')}
                </tr>`).join('')}</tbody>
            </table>
        </div>`;
    }
    const cr = data.classification_report || {};
    const pcHtml = `<div>
        <div style="font-family:monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:10px">Per-Class Report · Precision / Recall / F1 per star</div>
        <table style="width:100%;border-collapse:collapse">
            <thead><tr>
                ${['Star','Precision','Recall','F1-score','Support'].map((h, i) => `<th style="font-family:monospace;font-size:9.5px;color:#94a3b8;text-align:${i === 0 ? 'left' : 'right'};padding:8px 10px;border-bottom:1px solid #e2e8f0;font-weight:500">${h}</th>`).join('')}
            </tr></thead>
            <tbody>
                ${[1,2,3,4,5].map(s => {
                    const row = cr[s.toString()] || {};
                    const f1 = row['f1-score'] ?? 0;
                    return `<tr>
                        <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;padding:9px 10px;border-bottom:1px solid #f1f5f9">${s}★</td>
                        <td style="font-family:monospace;font-size:12px;color:#475569;text-align:right;padding:9px 10px;border-bottom:1px solid #f1f5f9">${(row.precision ?? 0).toFixed(3)}</td>
                        <td style="font-family:monospace;font-size:12px;color:#475569;text-align:right;padding:9px 10px;border-bottom:1px solid #f1f5f9">${(row.recall ?? 0).toFixed(3)}</td>
                        <td style="font-family:monospace;font-size:12px;color:#475569;text-align:right;padding:9px 10px;border-bottom:1px solid #f1f5f9">${f1.toFixed(3)}
                            <span style="display:inline-block;width:44px;height:6px;border-radius:99px;background:#f1f5f9;overflow:hidden;vertical-align:middle;margin-left:6px"><span style="display:block;height:100%;width:${(f1 * 100).toFixed(1)}%;background:${meta.color}"></span></span>
                        </td>
                        <td style="font-family:monospace;font-size:12px;color:#475569;text-align:right;padding:9px 10px;border-bottom:1px solid #f1f5f9">${Math.round(row.support ?? 0)}</td>
                    </tr>`;
                }).join('')}
                <tr style="background:#f8fafc">
                    <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;padding:9px 10px">Macro avg</td>
                    <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;text-align:right;padding:9px 10px">${(cr['macro avg']?.precision ?? 0).toFixed(3)}</td>
                    <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;text-align:right;padding:9px 10px">${(cr['macro avg']?.recall ?? 0).toFixed(3)}</td>
                    <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;text-align:right;padding:9px 10px">${data.f1_macro.toFixed(3)}</td>
                    <td style="font-family:monospace;font-size:12px;font-weight:600;color:#1e293b;text-align:right;padding:9px 10px">${Math.round(cr['macro avg']?.support ?? 0)}</td>
                </tr>
            </tbody>
        </table>
    </div>`;
    body.innerHTML = cmHtml + pcHtml;
}
// 6. Search Functionality
function initSearch() {
    const searchInput = document.getElementById('search-restaurant');
    const searchResults = document.getElementById('search-results');
    if (!searchInput || !searchResults)
        return;
    // Hide results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.add('hidden');
        }
    });
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (query.length < 1) {
            searchResults.classList.add('hidden');
            return;
        }
        // Filter restaurants by name or cuisine
        const filtered = allRestaurants.filter(r => r.name.toLowerCase().includes(query) ||
            r.cuisine.toLowerCase().includes(query));
        if (filtered.length === 0) {
            searchResults.innerHTML = `<div class="p-4 text-sm text-slate-500 text-center italic">No restaurants found</div>`;
            searchResults.classList.remove('hidden');
            return;
        }
        // Build results HTML
        let html = '';
        filtered.forEach(r => {
            html += `
                <div class="px-4 py-3 hover:bg-slate-50 cursor-pointer border-b border-slate-100 last:border-0 transition-all duration-300" onclick="focusRestaurantFromSearch(${r.id}, ${r.lat}, ${r.lng})">
                    <div class="flex flex-col gap-2">
                        <!-- Full-width Name Row -->
                        <div class="font-heading text-sm font-black text-slate-900 leading-snug w-full">
                            ${r.name}
                        </div>
                        
                        <!-- Bottom Meta Row -->
                        <div class="flex items-center justify-between w-full">
                            <span class="text-[9px] font-extrabold uppercase tracking-widest bg-brand-cyan/10 text-brand-cyan px-2 py-0.5 rounded-md border border-brand-cyan/20 whitespace-nowrap">
                                ${r.cuisine}
                            </span>
                            
                            <div class="flex items-center gap-2.5 text-[10px] font-bold text-slate-500">
                                <span class="flex items-center gap-1"><i class="fa-solid fa-user text-amber-500"></i> ${r.avg_user_rating.toFixed(1)}</span>
                                <span class="flex items-center gap-1"><i class="fa-solid fa-robot text-brand-cyan"></i> ${r.avg_ai_rating.toFixed(1)}</span>
                                ${r.has_anomaly ? `<span class="text-brand-red bg-rose-50 px-1.5 py-0.5 rounded border border-rose-100 flex items-center gap-1 ml-0.5"><i class="fa-solid fa-triangle-exclamation"></i> Alert</span>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        searchResults.innerHTML = html;
        searchResults.classList.remove('hidden');
    });
    // Auto-select first result on Enter key or Search button click
    const submitSearch = (e) => {
        if (e)
            e.preventDefault();
        const query = searchInput.value.toLowerCase().trim();
        if (query.length < 1)
            return;
        const filtered = allRestaurants.filter(r => r.name.toLowerCase().includes(query) ||
            r.cuisine.toLowerCase().includes(query));
        if (filtered.length > 0) {
            const r = filtered[0];
            focusRestaurantFromSearch(r.id, r.lat, r.lng);
        }
    };
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitSearch(e);
        }
    });
    const searchBtn = document.getElementById('search-btn');
    if (searchBtn) {
        searchBtn.addEventListener('click', submitSearch);
    }
}
function focusRestaurantFromSearch(id, lat, lng) {
    const searchInput = document.getElementById('search-restaurant');
    const searchResults = document.getElementById('search-results');
    if (searchInput)
        searchInput.value = '';
    if (searchResults)
        searchResults.classList.add('hidden');
    // Switch to reputation tab if not already there
    if (currentTab !== 'reputation') {
        switchTab('reputation');
    }
    // Zoom and Select
    map.setView([lat, lng], 16, { animate: true });
    selectRestaurant(id);
}
// Bind event handlers to the global window scope to avoid ES module reference errors
window.switchTab = switchTab;
window.focusRestaurantFromSearch = focusRestaurantFromSearch;
