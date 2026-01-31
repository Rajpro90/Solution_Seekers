// Mock Data consistent with your video
const mockData = {
    location: "Ahmedabad, India",
    current_aqi: 149,
    status: "Unhealthy for Sensitive Groups",
    hourly: [140, 165, 201, 180, 160, 145],
    daily: [149, 155, 140, 135, 142, 138, 130],
    disaster: { type: "Heatwave", prob: "87%", time: "12:00–16:00" }
};
// Add to your initCharts function
const ctx = document.getElementById('hourlyChart').getContext('2d');
const gradient = ctx.createLinearGradient(0, 0, 0, 400);
gradient.addColorStop(0, 'rgba(0, 128, 128, 0.4)'); // Teal glow
gradient.addColorStop(1, 'rgba(0, 128, 128, 0)');

// Use 'gradient' as your dataset's backgroundColor
let hChart, dChart;

function showPage(pageId) {
    console.log("Navigating to:", pageId); // Check your browser console (F12) to see if this runs

    // 1. Get all page sections
    const pages = document.querySelectorAll('.page');
    
    // 2. Remove 'active' class from all pages
    pages.forEach(page => {
        page.classList.remove('active');
    });

    // 3. Add 'active' class to the target page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        window.scrollTo(0, 0); // Reset scroll to top
    } else {
        console.error("Page ID not found:", pageId);
    }

    // 4. Load charts only if we are on the details page
    if (pageId === 'details') {
        // Use a slight timeout to let the CSS 'display: block' take effect before Chart.js renders
        setTimeout(initCharts, 100);
    }
}

// Ensure the Home page loads by default when the browser starts
window.onload = () => {
    showPage('home');
};

function initCharts() {
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

    hChart = new Chart(document.getElementById('hourlyChart'), {
        type: 'line',
        data: {
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            datasets: [{
                data: mockData.hourly,
                borderColor: '#008080',
                backgroundColor: 'rgba(0, 128, 128, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: commonOptions
    });

    dChart = new Chart(document.getElementById('dailyChart'), {
        type: 'line',
        data: {
            labels: ['Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu'],
            datasets: [{
                data: mockData.daily,
                borderColor: '#008080',
                tension: 0.4
            }]
        },
        options: commonOptions
    });
}

function fetchData() {
    // Logic to toggle between Mock and API would go here
    console.log("Data refreshed or Mode toggled");
}

// Start on Home
window.onload = () => showPage('home');
// Data for Live Mode (Simulating API response)
const liveData = {
    location: "Ahmedabad, India",
    current_aqi: 166, // Different value for live
    status: "Unhealthy",
    hourly: [150, 170, 195, 210, 185, 155],
    daily: [166, 160, 155, 150, 145, 140, 135],
    disaster: { type: "None", prob: "12%", time: "N/A" }
};

async function handleToggle() {
    const isChecked = document.getElementById('modeSwitch').checked;
    const statusText = document.getElementById('mode-status');

    if (isChecked) {
        statusText.innerText = "Test Mode";
        updateUI(mockData);
    } else {
        statusText.innerText = "Live API";
        try {
            // Fetch real data from your Python backend
            const response = await fetch('/api/live-data');
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error("Failed to fetch live data:", error);
            statusText.innerText = "Connection Error";
        }
    }
}

function updateUI(data) {
    // Dynamic background colors based on AQI (Like the video)
    const card = document.querySelector('.aqi-card-premium');
    if (data.current_aqi <= 50) {
        card.style.background = 'linear-gradient(145deg, #2ecc71, #27ae60)';
    } else if (data.current_aqi <= 100) {
        card.style.background = 'linear-gradient(145deg, #f1c40f, #f39c12)';
    } else {
        card.style.background = 'linear-gradient(145deg, #e67e22, #d35400)';
    }

    document.getElementById('aqi-num').innerText = data.current_aqi;
    document.getElementById('aqi-status').innerHTML = `<span class="warn-icon">⚠️</span> <strong>${data.status}</strong>`;
}