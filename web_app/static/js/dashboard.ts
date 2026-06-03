// TypeScript Definitions and Declarations for Leaflet.js
declare const L: any;

interface Review {
    restaurant_id: number;
    text: string;
    user_rating: number;
    ai_expected_rating: number;
    ai_hex_color: string;
    delta: number;
    is_anomaly: boolean;
}

interface Restaurant {
    id: number;
    name: string;
    lat: number;
    lng: number;
    cuisine: string;
    address: string;
    description: string;
    avg_user_rating: number;
    avg_ai_rating: number;
    review_count: number;
    anomaly_count: number;
    has_anomaly: boolean;
    ai_hex_color: string;
    reviews?: Review[];
}

interface EvaluationModelMetrics {
    n_samples: number;
    mae: number;
    rmse: number;
    accuracy: number;
    f1_macro: number;
    f1_weighted: number;
    per_class_recall: Record<string, number>;
    classification_report: Record<string, { "f1-score": number; recall: number }>;
    confusion_matrix: number[][];
}

interface EvaluationResults {
    models: {
        baseline?: EvaluationModelMetrics;
        xlmr?: EvaluationModelMetrics;
    };
}

interface EvaluationStatus {
    is_evaluating: boolean;
    progress: string;
}

// Global Variables
let map: any;
let markersGroup: any;
let currentTab: 'reputation' | 'evaluation' = 'reputation';
let evaluationPollInterval: number | null = null;
let allRestaurants: Restaurant[] = [];

// Initialize Dashboard on Load
window.addEventListener('DOMContentLoaded', () => {
    initMap();
    fetchRestaurants();
    checkExistingEvaluation();
    initSearch();
});

// 1. Map Initialization
function initMap(): void {
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
async function fetchRestaurants(): Promise<void> {
    try {
        const response = await fetch('/api/restaurants');
        if (!response.ok) throw new Error('Failed to load restaurants');
        const restaurants: Restaurant[] = await response.json();
        
        allRestaurants = restaurants;
        plotPins(restaurants);
    } catch (err) {
        console.error('Error fetching restaurants:', err);
    }
}

function plotPins(restaurants: Restaurant[]): void {
    markersGroup.clearLayers();
    
    restaurants.forEach((r: Restaurant) => {
        // Create custom neon circular pin HTML
        const customPinHTML = `
            <div class="custom-map-pin" style="
                background-color: ${r.ai_hex_color};
                border-color: ${r.has_anomaly ? '#e53935' : '#ffffff'};
                box-shadow: 0 0 15px ${r.ai_hex_color}, 0 0 5px rgba(0,0,0,0.5);
            "></div>
        `;
        
        const pinIcon = L.divIcon({
            html: customPinHTML,
            className: 'custom-pin-container',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
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
async function selectRestaurant(id: number): Promise<void> {
    // Show details section and hide empty placeholder state
    const placeholder = document.getElementById('reputation-placeholder');
    if (placeholder) placeholder.classList.add('hidden');
    
    const detailsContainer = document.getElementById('reputation-details');
    if (detailsContainer) detailsContainer.classList.remove('hidden');
    
    try {
        const response = await fetch(`/api/restaurants/${id}`);
        if (!response.ok) throw new Error('Failed to fetch restaurant details');
        const r: Restaurant = await response.json();
        
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
        
        if (restName) restName.innerText = r.name;
        if (restCuisine) restCuisine.innerText = r.cuisine;
        if (restAddress) restAddress.innerHTML = `<i class="fa-solid fa-location-dot"></i> ${r.address}`;
        if (restDesc) restDesc.innerText = r.description;
        
        // Ratings stats
        if (userRatingVal) userRatingVal.innerText = r.avg_user_rating.toFixed(2);
        if (aiRatingVal) aiRatingVal.innerText = r.avg_ai_rating.toFixed(2);
        
        // Stars rendering
        if (userRatingStars) userRatingStars.innerHTML = getStarRatingHTML(r.avg_user_rating);
        if (aiRatingStars) aiRatingStars.innerHTML = getStarRatingHTML(r.avg_ai_rating);
        
        // Counters
        if (totalReviewsCount) totalReviewsCount.innerText = r.review_count.toString();
        if (anomaliesCount) anomaliesCount.innerText = r.anomaly_count.toString();
        
        // Anomaly warnings
        const fraudBanner = document.getElementById('fraud-alert-banner');
        if (fraudBanner) {
            if (r.has_anomaly) {
                fraudBanner.classList.remove('hidden');
            } else {
                fraudBanner.classList.add('hidden');
            }
        }
        
        // Render detailed reviews list logs
        if (r.reviews) {
            renderReviewLogs(r.reviews);
        }
        
    } catch (err) {
        console.error('Error loading details:', err);
    }
}

// Helper to draw stars dynamically based on ratings (out of 5 stars)
function getStarRatingHTML(rating: number): string {
    let stars = '';
    const rounded = Math.round(rating * 2) / 2; // Round to nearest 0.5
    
    for (let i = 1; i <= 5; i++) {
        if (i <= rounded) {
            stars += '<i class="fa-solid fa-star"></i>';
        } else if (i - 0.5 === rounded) {
            stars += '<i class="fa-solid fa-star-half-stroke"></i>';
        } else {
            stars += '<i class="fa-regular fa-star"></i>';
        }
    }
    return stars;
}

// Render Review Log List Cards
function renderReviewLogs(reviews: Review[]): void {
    const listContainer = document.getElementById('reviews-list-container');
    if (!listContainer) return;
    listContainer.innerHTML = ''; // Clear logs
    
    reviews.forEach((rev: Review) => {
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
function switchTab(tabName: 'reputation' | 'evaluation'): void {
    currentTab = tabName;
    
    // Active tabs class swap
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    const tabBtn = document.getElementById(`tab-btn-${tabName}`);
    const tabContent = document.getElementById(`tab-content-${tabName}`);
    if (tabBtn) tabBtn.classList.add('active');
    if (tabContent) tabContent.classList.add('active');
    
    // Map invalidateSize to avoid Leaflet rendering bugs inside dynamic hidden tabs
    if (tabName === 'reputation') {
        setTimeout(() => map.invalidateSize(), 50);
    }
}

// 5. Dual Model Evaluation Dashboard Flow
async function checkExistingEvaluation(): Promise<void> {
    try {
        const response = await fetch('/api/evaluation-results');
        if (response.ok) {
            const results: EvaluationResults = await response.json();
            displayEvaluationResults(results);
        }
    } catch (err) {
        // Safe to ignore on startup if no evaluation file exists
    }
}

async function triggerEvaluation(): Promise<void> {
    const btn = document.getElementById('btn-run-eval') as HTMLButtonElement | null;
    const progressContainer = document.getElementById('eval-progress-container');
    const progressBar = document.getElementById('eval-progress-bar');
    const progressText = document.getElementById('eval-status-text');
    
    if (btn) btn.disabled = true;
    if (progressContainer) progressContainer.classList.remove('hidden');
    if (progressBar) progressBar.style.width = '10%';
    if (progressText) progressText.innerText = 'Triggering dual comparison pipeline...';
    
    try {
        const response = await fetch('/api/evaluate', { method: 'POST' });
        if (!response.ok) throw new Error('Failed to start evaluation');
        
        // Start polling
        startEvaluationPolling();
    } catch (err: any) {
        if (progressText) progressText.innerText = 'Trigger failed: ' + err.message;
        if (btn) btn.disabled = false;
    }
}

function startEvaluationPolling(): void {
    if (evaluationPollInterval) clearInterval(evaluationPollInterval);
    
    const progressBar = document.getElementById('eval-progress-bar');
    const progressText = document.getElementById('eval-status-text');
    const btn = document.getElementById('btn-run-eval') as HTMLButtonElement | null;
    
    let simulatedProgress = 10;
    
    evaluationPollInterval = window.setInterval(async () => {
        try {
            const response = await fetch('/api/evaluation-status');
            if (!response.ok) return;
            const status: EvaluationStatus = await response.json();
            
            // Advance progress simulation
            if (simulatedProgress < 90) {
                simulatedProgress += 10;
                if (progressBar) progressBar.style.width = `${simulatedProgress}%`;
            }
            
            if (progressText) progressText.innerText = status.progress;
            
            if (!status.is_evaluating) {
                if (evaluationPollInterval) clearInterval(evaluationPollInterval);
                if (progressBar) progressBar.style.width = '100%';
                
                if (status.progress.includes('Failed')) {
                    if (progressText) progressText.innerText = 'Evaluation failed!';
                    if (btn) btn.disabled = false;
                } else {
                    if (progressText) progressText.innerText = 'Evaluation completed! Building Plotly dashboard...';
                    setTimeout(async () => {
                        // Fetch final metrics and render charts
                        const resultsResponse = await fetch('/api/evaluation-results');
                        if (resultsResponse.ok) {
                            const results: EvaluationResults = await resultsResponse.json();
                            displayEvaluationResults(results);
                        }
                        
                        // Hide progress bar and enable button
                        const progContainer = document.getElementById('eval-progress-container');
                        if (progContainer) progContainer.classList.add('hidden');
                        if (btn) btn.disabled = false;
                        
                        // Refresh pins to update potential color-mappings
                        fetchRestaurants();
                    }, 1000);
                }
            }
        } catch (err) {
            console.error('Error polling evaluation status:', err);
        }
    }, 2000);
}

// Display final evaluation stats cards and Plotly HTML iframe
function displayEvaluationResults(results: EvaluationResults): void {
    const emptyState = document.getElementById('evaluation-empty-state');
    if (emptyState) emptyState.classList.add('hidden');
    
    const resultsDashboard = document.getElementById('evaluation-results-dashboard');
    if (resultsDashboard) resultsDashboard.classList.remove('hidden');
    
    const models = results.models;
    if (models.baseline && models.xlmr) {
        // Set metrics on screen
        const mBaseAcc = document.getElementById('m-base-acc');
        const mAdvAcc = document.getElementById('m-adv-acc');
        const mBaseMae = document.getElementById('m-base-mae');
        const mAdvMae = document.getElementById('m-adv-mae');
        
        if (mBaseAcc) mBaseAcc.innerText = `${(models.baseline.accuracy * 100).toFixed(1)}%`;
        if (mAdvAcc) mAdvAcc.innerText = `${(models.xlmr.accuracy * 100).toFixed(1)}%`;
        
        if (mBaseMae) mBaseMae.innerText = `${models.baseline.mae.toFixed(3)}★`;
        if (mAdvMae) mAdvMae.innerText = `${models.xlmr.mae.toFixed(3)}★`;
        
        // Set interactive Plotly iframe report source
        const iframe = document.getElementById('eval-plotly-iframe') as HTMLIFrameElement | null;
        if (iframe) iframe.src = '/api/evaluation-viz?t=' + new Date().getTime(); // Anti-caching query string
    }
}

// 6. Search Functionality
function initSearch(): void {
    const searchInput = document.getElementById('search-restaurant') as HTMLInputElement | null;
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    // Hide results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target as Node) && !searchResults.contains(e.target as Node)) {
            searchResults.classList.add('hidden');
        }
    });
    
    searchInput.addEventListener('input', (e) => {
        const query = (e.target as HTMLInputElement).value.toLowerCase().trim();
        
        if (query.length < 1) {
            searchResults.classList.add('hidden');
            return;
        }
        
        // Filter restaurants by name or cuisine
        const filtered = allRestaurants.filter(r => 
            r.name.toLowerCase().includes(query) || 
            r.cuisine.toLowerCase().includes(query)
        );
        
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
    const submitSearch = (e?: Event) => {
        if (e) e.preventDefault();
        const query = searchInput.value.toLowerCase().trim();
        if (query.length < 1) return;
        
        const filtered = allRestaurants.filter(r => 
            r.name.toLowerCase().includes(query) || 
            r.cuisine.toLowerCase().includes(query)
        );
        
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

function focusRestaurantFromSearch(id: number, lat: number, lng: number): void {
    const searchInput = document.getElementById('search-restaurant') as HTMLInputElement | null;
    const searchResults = document.getElementById('search-results');
    
    if (searchInput) searchInput.value = '';
    if (searchResults) searchResults.classList.add('hidden');
    
    // Switch to reputation tab if not already there
    if (currentTab !== 'reputation') {
        switchTab('reputation');
    }
    
    // Zoom and Select
    map.setView([lat, lng], 16, { animate: true });
    selectRestaurant(id);
}

// Bind event handlers to the global window scope to avoid ES module reference errors
(window as any).switchTab = switchTab;
(window as any).triggerEvaluation = triggerEvaluation;
(window as any).focusRestaurantFromSearch = focusRestaurantFromSearch;
