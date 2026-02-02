// Dataset 1: Dummy/Test Data
const mockData = {
    location: "Ahmedabad (Test)",
    current_aqi: 149,
    status: "Unhealthy for Sensitive Groups",
    hourly_details: [
        { time: "06:00", aqi: 155, status: "Unhealthy" },
        { time: "09:00", aqi: 186, status: "Unhealthy" },
        { time: "12:00", aqi: 149, status: "Sensitive" }
    ],
    daily_details: [
        { date: "Feb 06", aqi: 154, trend: "down" },
        { date: "Feb 07", aqi: 150, trend: "down" }
    ],
    hourly: [155, 186, 149, 140, 135, 130, 145],
    daily: [154, 150, 147, 140, 135, 130, 125]
    , disaster: {
        type: "Heatwave",
        prob: "87%",
        time: "12:00–16:00",
        date: "February 06, 2026" // Added Date
    }
};
// Dataset 2: Simulated "Live" Data (Used if backend fails or for preview)
const simulatedLiveData = {
    location: "Mumbai (Live)",
    current_aqi: 82,
    status: "Moderate",
    hourly_details: [
        { time: "06:00", aqi: 70, status: "Good" },
        { time: "09:00", aqi: 85, status: "Moderate" },
        { time: "12:00", aqi: 82, status: "Moderate" }
    ],
    daily_details: [
        { date: "Feb 06", aqi: 82, trend: "up" },
        { date: "Feb 07", aqi: 88, trend: "up" }
    ],
    hourly: [70, 85, 82, 90, 95, 100, 80],
    daily: [82, 88, 92, 95, 100, 105, 110]
    , disaster: {
        type: "None",
        prob: "5%",
        time: "N/A",
        date: "February 02, 2026" // Added Date
    }
};

let hChart, dChart;
let currentActiveData = mockData; // Track current data globally

// NAVIGATION
function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));

    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        window.scrollTo(0, 0);
    }

    if (pageId === 'details') {
        setTimeout(() => {
            updateTables(currentActiveData);
            initCharts(currentActiveData);
        }, 100);
    }
}

// TOGGLE LOGIC
async function handleToggle() {
    const isChecked = document.getElementById('modeSwitch').checked; // Test Mode is ON when checked
    const statusText = document.getElementById('mode-status');
    const badge = document.querySelector('.ai-label');

    if (isChecked) {
        // --- TEST MODE ON ---
        statusText.innerHTML = "Test Mode";
        statusText.style.color = "#8b949e";
        badge.innerHTML = "🤖 AI-Powered Climate Insights";

        updateUI(mockData);
    } else {
        // --- LIVE MODE ON ---
        // ... inside handleToggle() else block ...
        const response = await fetch('http://127.0.0.1:5000/api/predict/aqi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: 23.0225, longitude: 72.5714 })
        });

        const backendData = await response.json();

        // TRANSLATION LAYER: Map backend keys to frontend keys
        currentActiveData = {
            location: backendData.location,
            current_aqi: backendData.forecast[0].aqi,
            status: backendData.forecast[0].status,
            // Convert backend 'forecast' to 'hourly_details'
            hourly_details: backendData.forecast.slice(0, 6).map((item, i) => ({
                time: `${i * 4}:00`, // Creating time slots
                aqi: item.aqi,
                status: item.status
            })),
            // Convert backend 'forecast' to 'daily_details'
            daily_details: backendData.forecast.map(item => ({
                date: item.date,
                aqi: item.aqi,
                trend: "stable"
            })),
            hourly: backendData.forecast.slice(0, 6).map(item => item.aqi),
            daily: backendData.forecast.map(item => item.aqi),
            disaster: {
                type: backendData.disaster_risk || "None",
                prob: "80%",
                time: "10:00 - 18:00",
                date: backendData.forecast[0].date
            }
        };

        updateUI(currentActiveData);
    }
}

async function fetchLiveData(lat, lon) {
    try {
        const response = await fetch('http://127.0.0.1:5000/api/predict/aqi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lon })
        });

        if (!response.ok) throw new Error("Backend Error");

        const apiData = await response.json();

        // Map Backend Response to Frontend Format
        const mappedData = {
            location: apiData.location.city || "Unknown City",
            current_aqi: apiData.location.current_aqi || 0,
            status: apiData.location.current_aqi > 150 ? "Unhealthy" : "Good", // Simple logic, can be enhanced
            hourly_details: [], // Backend currently simulates hourly, can add endpoint later
            daily_details: apiData.forecast.map(item => ({
                date: new Date(item.Date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                aqi: item['Predicted AQI'],
                trend: item['Predicted AQI'] > apiData.location.current_aqi ? 'up' : 'down'
            })),
            hourly: [apiData.location.current_aqi, ...apiData.forecast.slice(0, 5).map(x => x['Predicted AQI'])], // Fallback visualization
            daily: apiData.forecast.map(x => x['Predicted AQI'])
        };

        // Populate mock hourly if empty
        mappedData.hourly_details = [
            { time: "Now", aqi: mappedData.current_aqi, status: "Current" },
            { time: "+1 Day", aqi: mappedData.daily[0], status: "Forecast" },
            { time: "+2 Days", aqi: mappedData.daily[1], status: "Forecast" }
        ];

        currentActiveData = mappedData;
        updateUI(currentActiveData);

    } catch (error) {
        console.error("Error fetching live data:", error);
        alert("Failed to connect to backend. Make sure 'python backend/run.py' is running.");
        // Revert UI to show error state or keep previous
    }
}

// UI UPDATES
function updateUI(data) {
    const aqiDisplay = document.getElementById('aqi-num');
    aqiDisplay.style.opacity = "0"; // Fade out

    setTimeout(() => {
        aqiDisplay.innerText = data.current_aqi;
        aqiDisplay.style.opacity = "1"; // Fade back in with new data
    }, 200);
    document.getElementById('city-name').innerText = data.location;
    document.getElementById('aqi-num').innerText = data.current_aqi;

    // Update color based on AQI
    const card = document.querySelector('.aqi-card-premium');
    if (data.current_aqi > 150) {
        card.style.background = 'linear-gradient(145deg, #e67e22, #d35400)';
    } else {
        card.style.background = 'linear-gradient(145deg, #008080, #006666)';
    }

    // Refresh tables and charts if on details page
    if (document.getElementById('details').classList.contains('active')) {
        updateTables(data);
        initCharts(data);
    }
    if (data.disaster) {
        document.getElementById('disaster-type').innerText = data.disaster.type;
        document.getElementById('disaster-date').innerText = data.disaster.date; // Updates the date
        document.getElementById('disaster-prob').innerText = data.disaster.prob;
        document.getElementById('disaster-time').innerText = data.disaster.time;

        // Update the Home Page preview strip too
        document.getElementById('disaster-preview').innerText =
            `${data.disaster.type} predicted for ${data.disaster.date}`;
    }
    document.getElementById('city-name').innerText = data.location;
    document.getElementById('aqi-num').innerText = data.current_aqi;

    // Update Disaster Section with Date
    if (data.disaster) {
        document.getElementById('disaster-type').innerText = data.disaster.type;
        // Fix for Disaster Subtitle Date
        const dateEl = document.querySelector('.disaster-subtitle');
        if (dateEl) dateEl.innerHTML = `Predicted for: <span style="color:white; font-weight:bold">${data.disaster.date}</span>`;

        document.getElementById('disaster-prob').innerText = data.disaster.prob;
        document.getElementById('disaster-time').innerText = data.disaster.time;
    }

    updateTables(data);
    initCharts(data);
}

// Helper to generate 24-hour slots (00:00, 04:00, etc.)
function generateTimeSlots(count) {
    let slots = [];
    for (let i = 0; i < count; i++) {
        let hour = (i * 4) % 24; 
        slots.push(`${hour.toString().padStart(2, '0')}:00`);
    }
    return slots;
}

function updateTables(data) {
    const hBody = document.getElementById('hourly-table-body');
    const dBody = document.getElementById('daily-table-body');

    // FIX: 24-Hour Forecast Table (Time Format)
    if (hBody && data.hourly_details) {
        const times = generateTimeSlots(data.hourly_details.length);
        hBody.innerHTML = data.hourly_details.map((item, index) => `
            <tr>
                <td>${times[index]}</td> <td style="font-weight:bold; color:#f39c12">${Math.round(item.aqi)}</td>
                <td><span class="status-tag">${item.status || 'Analyzed'}</span></td>
            </tr>
        `).join('');
    }

    // Daily Table (7-Day Forecast)
    if (dBody && data.daily_details) {
        dBody.innerHTML = data.daily_details.map(item => `
            <tr>
                <td>${item.date}</td>
                <td style="font-weight:bold; color:#008080">${Math.round(item.aqi)}</td>
                <td>${item.trend === 'up' ? '📈 Rising' : '📉 Falling'}</td>
            </tr>
        `).join('');
    }
}

// Update the updateUI to include the Disaster Date
function updateUI(data) {
    document.getElementById('city-name').innerText = data.location;
    document.getElementById('aqi-num').innerText = data.current_aqi;

    // Update Disaster Section with Date
    if (data.disaster) {
        document.getElementById('disaster-type').innerText = data.disaster.type;
        // Fix for Disaster Subtitle Date
        const dateEl = document.querySelector('.disaster-subtitle');
        if (dateEl) dateEl.innerHTML = `Predicted for: <span style="color:white; font-weight:bold">${data.disaster.date}</span>`;

        document.getElementById('disaster-prob').innerText = data.disaster.prob;
        document.getElementById('disaster-time').innerText = data.disaster.time;
    }

    updateTables(data);
    initCharts(data);
}

// Helper to match ML status
function getAQIStatus(aqi) {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    return "Unhealthy";
}
function initCharts(data) {
    // backgroundColor: gradient
    if (hChart) hChart.destroy();
    if (dChart) dChart.destroy();

    const hCtx = document.getElementById('hourlyChart').getContext('2d');
    
    // Create a smooth gradient for the area under the line
    const gradient = hCtx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 128, 128, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 128, 128, 0)');

    hChart = new Chart(hCtx, {
        type: 'line',
        data: {
            // 7 labels to cover: 0, 4, 8, 12, 16, 20, and the start of next day (24)
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
            datasets: [{
                label: 'AQI Forecast',
                data: data.hourly, // Ensure this array has 7 numbers
                borderColor: '#008080',
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 8,
                pointBackgroundColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    grid: { display: false }, 
                    ticks: { color: '#8b949e' } 
                },
                y: { 
                    beginAtZero: false,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
                    ticks: { color: '#8b949e' } 
                }
            }
        }
    });

    const dCtx = document.getElementById('dailyChart').getContext('2d');
    dChart = new Chart(dCtx, {
        type: 'line',
        data: {
            labels: ['Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu'],
            datasets: [{
                data: data.daily,
                borderColor: '#008080',
                tension: 0.4
            }]
        },
        options: commonOptions
    });
}

// STARTUP
window.onload = () => {
    showPage('home');
    updateUI(mockData);
};
