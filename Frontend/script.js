// Dataset 1: Dummy/Test Data
const mockData = {
    location: "Ahmedabad (Test)",
    current_aqi: 149,
    status: "Unhealthy for Sensitive Groups",
    hourly_details: [
        {time: "06:00", aqi: 155, status: "Unhealthy"},
        {time: "09:00", aqi: 186, status: "Unhealthy"},
        {time: "12:00", aqi: 149, status: "Sensitive"}
    ],
    daily_details: [
        {date: "Feb 06", aqi: 154, trend: "down"},
        {date: "Feb 07", aqi: 150, trend: "down"}
    ],
    hourly: [155, 186, 149, 140, 135, 130],
    daily: [154, 150, 147, 140, 135, 130, 125]
};

// Dataset 2: Simulated "Live" Data (Used if backend fails or for preview)
const simulatedLiveData = {
    location: "Mumbai (Live)",
    current_aqi: 82,
    status: "Moderate",
    hourly_details: [
        {time: "06:00", aqi: 70, status: "Good"},
        {time: "09:00", aqi: 85, status: "Moderate"},
        {time: "12:00", aqi: 82, status: "Moderate"}
    ],
    daily_details: [
        {date: "Feb 06", aqi: 82, trend: "up"},
        {date: "Feb 07", aqi: 88, trend: "up"}
    ],
    hourly: [70, 85, 82, 90, 95, 100],
    daily: [82, 88, 92, 95, 100, 105, 110]
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
    const isTestMode = document.getElementById('modeSwitch').checked;
    const statusLabel = document.getElementById('mode-status');
    
    if (isTestMode) {
        statusLabel.innerText = "Test Mode";
        statusLabel.style.color = "#8b949e";
        currentActiveData = mockData;
        updateUI(currentActiveData);
    } else {
        statusLabel.innerText = "Live API";
        statusLabel.style.color = "#4ade80";
        try {
            const response = await fetch('http://127.0.0.1:5000/api/live-data');
            if (!response.ok) throw new Error("Offline");
            currentActiveData = await response.json();
        } catch (error) {
            console.log("Using simulated live data (No backend found)");
            currentActiveData = simulatedLiveData;
        }
        updateUI(currentActiveData);
    }
}

// UI UPDATES
function updateUI(data) {
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
}

function updateTables(data) {
    const hBody = document.getElementById('hourly-table-body');
    const dBody = document.getElementById('daily-table-body');

    if (hBody && data.hourly_details) {
        hBody.innerHTML = data.hourly_details.map(item => `
            <tr>
                <td>${item.time}</td>
                <td style="font-weight:bold; color:#f39c12">${item.aqi}</td>
                <td><span class="status-tag">${item.status}</span></td>
            </tr>
        `).join('');
    }

    if (dBody && data.daily_details) {
        dBody.innerHTML = data.daily_details.map(item => `
            <tr>
                <td>${item.date}</td>
                <td style="font-weight:bold; color:#008080">${item.aqi}</td>
                <td>${item.trend === 'up' ? '📈 Rising' : '📉 Falling'}</td>
            </tr>
        `).join('');
    }
}

function initCharts(data) {
    if (hChart) hChart.destroy();
    if (dChart) dChart.destroy();

    const commonOptions = {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { 
            x: { grid: { display: false }, ticks: { color: '#8b949e' } },
            y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } }
        }
    };

    const hCtx = document.getElementById('hourlyChart').getContext('2d');
    hChart = new Chart(hCtx, {
        type: 'line',
        data: {
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            datasets: [{
                data: data.hourly,
                borderColor: '#008080',
                backgroundColor: 'rgba(0, 128, 128, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: commonOptions
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
