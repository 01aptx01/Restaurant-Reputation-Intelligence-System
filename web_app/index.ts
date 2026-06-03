import { Elysia } from "elysia";
import { spawnSync, spawn } from "child_process";
import { join } from "path";

const app = new Elysia();

// Define 8 Famous Restaurants in Bangkok
const BANGKOK_RESTAURANTS = [
    {
        id: 1,
        name: "Jay Fai (เจ๊ไฝ)",
        lat: 13.7526,
        lng: 100.5048,
        cuisine: "Thai Seafood & Street Food",
        address: "327 Mahachai Rd, Samran Rat, Phra Nakhon, Bangkok 10200",
        description: "Legendary Michelin-starred street food stall famous for its iconic crab omelet.",
    },
    {
        id: 2,
        name: "Sorn (ศรณ์)",
        lat: 13.7314,
        lng: 100.5696,
        cuisine: "Southern Thai Fine Dining",
        address: "56 Sukhumvit 26 Soi Ari, Khlong Toei, Bangkok 10110",
        description: "Refined Southern Thai recipes using ultra-premium local ingredients in a restored house.",
    },
    {
        id: 3,
        name: "Le Du (ฤดู)",
        lat: 13.7259,
        lng: 100.5284,
        cuisine: "Modern Thai Culinary",
        address: "399/3 Silom Soi 7, Bang Rak, Bangkok 10500",
        description: "Innovative agricultural-focused Thai dining using modern techniques.",
    },
    {
        id: 4,
        name: "Sühring (ซูห์ริง)",
        lat: 13.7128,
        lng: 100.5435,
        cuisine: "Modern German Fine Dining",
        address: "10 Soi Yen Akat 3, Chong Nonsi, Yan Nawa, Bangkok 10120",
        description: "Twin chefs offering childhood memories via high-end German gastronomical creations.",
    },
    {
        id: 5,
        name: "Gaggan Anand",
        lat: 13.7345,
        lng: 100.5732,
        cuisine: "Progressive Indian Cuisine",
        address: "68 Sukhumvit Soi 31, Khlong Toei Nuea, Watthana, Bangkok 10110",
        description: "An avant-garde, rebellious progressive Indian culinary theater.",
    },
    {
        id: 6,
        name: "Jeh O Chula (เจ๊โอว)",
        lat: 13.7428,
        lng: 100.5255,
        cuisine: "Late Night Thai Street Food",
        address: "113 Rong Muang, Pathum Wan, Bangkok 10330",
        description: "Famous for its massive late-night Tom Yum Mama noodles and crispy pork belly.",
    },
    {
        id: 7,
        name: "Somtum Der (ส้มตำเด้อ)",
        lat: 13.7275,
        lng: 100.5332,
        cuisine: "Authentic Isan / Northeastern",
        address: "5/5 Sala Daeng Rd, Silom, Bang Rak, Bangkok 10500",
        description: "Fiery and authentic Northeastern Isan food in a vibrant, casual atmosphere.",
    },
    {
        id: 8,
        name: "Paste Bangkok",
        lat: 13.7441,
        lng: 100.5401,
        cuisine: "Heirloom Royal Thai",
        address: "999 Phloen Chit Rd, Lumpini, Pathum Wan, Bangkok 10330",
        description: "Reinterpreted traditional royal Thai recipes with intricate flavor layers.",
    },
];

// Helper to determine star hex colors
function getHexColor(rating: number): string {
    const clamped = Math.max(1, Math.min(5, Math.round(rating)));
    const colors: Record<number, string> = {
        1: "#e53935",
        2: "#ff9800",
        3: "#fbc02d",
        4: "#4caf50",
        5: "#00bcd4",
    };
    return colors[clamped];
}

// Global cache for reviews and restaurants data
let reviewsCache: any[] = [];
let restaurantsCache: any[] = [];
let isEvaluating = false;
let evalProgress = "Idle";

// Load scored reviews from initializer
async function loadScoredReviews() {
    console.log("Loading scored reviews...");
    try {
        const file = Bun.file(join(__dirname, "scored_reviews.json"));
        if (await file.exists()) {
            reviewsCache = await file.json();
            
            // Extract unique place_names and build restaurants
            const placeNames = Array.from(new Set(reviewsCache.map(rev => rev.place_name).filter(Boolean)));
            if (placeNames.length > 0) {
                // We have data from 70k.tsv
                restaurantsCache = placeNames.map((name, idx) => {
                    const r_id = idx + 1;
                    const r_reviews = reviewsCache
                        .filter(rev => rev.place_name === name)
                        .map(rev => ({ ...rev, restaurant_id: r_id }));
                    
                    const user_ratings = r_reviews.map(rev => rev.user_rating);
                    const ai_ratings = r_reviews.map(rev => rev.ai_expected_rating);
                    const anomaly_count = r_reviews.filter(rev => rev.is_anomaly).length;
                    
                    const avg_user = user_ratings.reduce((a, b) => a + b, 0) / (user_ratings.length || 1);
                    const avg_ai = ai_ratings.reduce((a, b) => a + b, 0) / (ai_ratings.length || 1);
                    
                    // Generate random coords around Bangkok center for visualization
                    const lat = 13.75 + (Math.random() * 0.2 - 0.1);
                    const lng = 100.5 + (Math.random() * 0.2 - 0.1);

                    return {
                        id: r_id,
                        name: name as string,
                        lat,
                        lng,
                        cuisine: "General Food",
                        address: "Location in Dataset",
                        description: "Data loaded from dataset.",
                        reviews: r_reviews,
                        avg_user_rating: parseFloat(avg_user.toFixed(2)),
                        avg_ai_rating: parseFloat(avg_ai.toFixed(2)),
                        review_count: r_reviews.length,
                        anomaly_count,
                        has_anomaly: anomaly_count > 0,
                        ai_hex_color: getHexColor(avg_ai),
                    };
                });
                console.log(`Successfully extracted ${restaurantsCache.length} restaurants from dataset and distributed ${reviewsCache.length} reviews.`);
            } else {
                // Fallback to hardcoded BANGKOK_RESTAURANTS if place_name is missing
                restaurantsCache = BANGKOK_RESTAURANTS.map(r_meta => {
                    const r_id = r_meta.id;
                    const r_reviews = reviewsCache
                        .filter((_, i) => (i % BANGKOK_RESTAURANTS.length) + 1 === r_id)
                        .map(rev => ({ ...rev, restaurant_id: r_id }));
                    
                    const user_ratings = r_reviews.map(rev => rev.user_rating);
                    const ai_ratings = r_reviews.map(rev => rev.ai_expected_rating);
                    const anomaly_count = r_reviews.filter(rev => rev.is_anomaly).length;
                    
                    const avg_user = user_ratings.reduce((a, b) => a + b, 0) / (user_ratings.length || 1);
                    const avg_ai = ai_ratings.reduce((a, b) => a + b, 0) / (ai_ratings.length || 1);
                    
                    return {
                        ...r_meta,
                        reviews: r_reviews,
                        avg_user_rating: parseFloat(avg_user.toFixed(2)),
                        avg_ai_rating: parseFloat(avg_ai.toFixed(2)),
                        review_count: r_reviews.length,
                        anomaly_count,
                        has_anomaly: anomaly_count > 0,
                        ai_hex_color: getHexColor(avg_ai),
                    };
                });
                console.log(`Successfully distributed ${reviewsCache.length} reviews to 8 fallback restaurants.`);
            }
        } else {
            console.warn("scored_reviews.json not found! Running data initializer...");
            const pythonExe = join(__dirname, "..", ".venv", "Scripts", "python.exe");
            const initScript = join(__dirname, "initialize_data.py");
            const initProc = spawnSync(pythonExe, [initScript, "--model", "auto"]);
            if (initProc.status === 0) {
                await loadScoredReviews();
            } else {
                console.error("Data initialization failed!", initProc.stderr ? initProc.stderr.toString() : "");
            }
        }
    } catch (e) {
        console.error("Error loading scored reviews:", e);
    }
}

// Perform initial load
loadScoredReviews();

// Setup Elysia Server
app
    // Serves the HTML frontend shell
    .get("/", () => Bun.file(join(__dirname, "templates", "index.html")))
    
    // Serve Static scripts and stylesheets
    .get("/static/js/dashboard.js", () => Bun.file(join(__dirname, "static", "js", "dashboard.js")))
    .get("/static/css/style.css", () => Bun.file(join(__dirname, "static", "css", "style.css")))
    
    // API: List all restaurants (without nesting heavy review arrays to keep payload light)
    .get("/api/restaurants", () => {
        return restaurantsCache.map(r => {
            const { reviews, ...rest } = r;
            return rest;
        });
    })
    
    // API: Fetch individual restaurant reviews and details
    .get("/api/restaurants/:id", ({ params }) => {
        const id = parseInt(params.id);
        const r = restaurantsCache.find(item => item.id === id);
        if (!r) {
            return new Response("Restaurant not found", { status: 404 });
        }
        return r;
    })
    
    // API: Trigger dual-model evaluation asynchronously
    .post("/api/evaluate", () => {
        if (isEvaluating) {
            return { status: "already_running", progress: evalProgress };
        }
        
        isEvaluating = true;
        evalProgress = "Scheduling evaluation on all 3 models...";
        
        const projectRoot = join(__dirname, "..");
        const pythonExe = join(projectRoot, ".venv", "Scripts", "python.exe");
        const reportPath = join(projectRoot, "outputs", "eval", "eval_report.json");
        
        // Spawn evaluation background process for all models
        const proc = spawn(
            pythonExe,
            ["-m", "rris", "evaluate", "--model", "all", "--output", reportPath],
            { cwd: projectRoot }
        );
        
        proc.stdout?.on("data", (data) => {
            const text = data.toString();
            if (text.includes("Baseline") || text.includes("XGBoost")) {
                evalProgress = "Evaluating Baseline (TF-IDF + XGBoost)...";
            } else if (text.includes("XLM-RoBERTa") || text.includes("XLM-R")) {
                evalProgress = "Evaluating XLM-R (this may take a minute)...";
            } else if (text.includes("Embedding") || text.includes("Sentence Embedding")) {
                evalProgress = "Evaluating Sentence Embedding...";
            }
        });
        
        proc.on("close", async (code) => {
            if (code !== 0) {
                isEvaluating = false;
                evalProgress = "Failed: rris evaluate exited with error code " + code;
                return;
            }
            
            try {
                evalProgress = "Generating Plotly comparative visualizations...";
                const vizPath = join(projectRoot, "outputs", "reports", "eval_report_viz.html");
                
                const vizProc = spawn(
                    pythonExe,
                    [
                        "-m", "rris", "visualize",
                        "--input", reportPath,
                        "--output", vizPath,
                    ],
                    { cwd: projectRoot }
                );
                
                vizProc.on("close", async (codeViz) => {
                    if (codeViz === 0) {
                        evalProgress = "Refreshing map with best model...";
                        const initScript = join(__dirname, "initialize_data.py");
                        const initProc = spawnSync(pythonExe, [initScript, "--model", "auto"], {
                            cwd: projectRoot,
                        });
                        if (initProc.status === 0) {
                            await loadScoredReviews();
                            evalProgress = "Completed";
                        } else {
                            evalProgress = "Eval done; map refresh failed";
                        }
                    } else {
                        evalProgress = "Failed: rris visualize exited with code " + codeViz;
                    }
                    isEvaluating = false;
                });
            } catch (err: any) {
                isEvaluating = false;
                evalProgress = "Failed: " + err.message;
            }
        });
        
        return { status: "started", progress: "Evaluation pipeline scheduled." };
    })
    
    // API: Fetch evaluation status
    .get("/api/evaluation-status", () => {
        return { is_evaluating: isEvaluating, progress: evalProgress };
    })
    
    // API: Fetch evaluation results JSON
    .get("/api/evaluation-results", async () => {
        const reportPath = join(__dirname, "..", "outputs", "eval", "eval_report.json");
        const file = Bun.file(reportPath);
        if (await file.exists()) {
            return file;
        }
        return new Response("Evaluation report not found", { status: 404 });
    })
    
    // API: Fetch Plotly Comparative Visual Report
    .get("/api/evaluation-viz", async () => {
        const vizPath = join(__dirname, "..", "outputs", "reports", "eval_report_viz.html");
        const file = Bun.file(vizPath);
        if (await file.exists()) {
            return file;
        }
        return new Response("Visualization not found", { status: 404 });
    });

// Listen on 127.0.0.1:8000
app.listen({ port: 8000, hostname: "127.0.0.1" }, () => {
    console.log("Bun + Elysia server running at http://127.0.0.1:8000");
});