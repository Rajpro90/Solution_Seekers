// async function saveLocationToDB(aqiData, cityName) {
//     const payload = {
//         city: cityName,
//         lat: aqiData.regions[0].lat,
//         lng: aqiData.regions[0].lng,
//         overall_aqi: aqiData.overall_aqi,
//         category: aqiData.category,
//         hourly: aqiData.regions[0].hourly
//     };

//     try {
//         const response = await fetch('http://127.0.0.1:5000/save_location', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(payload)
//         });
//         const result = await response.json();
//         console.log(result.message);
//     } catch (error) {
//         console.error("Error saving to database:", error);
//     }
// }
// // -------------------- CONFIG & DATA --------------------
// // 1. Data for testing
// const dummyAQIData = {
//     good: { overall_aqi: 25, category: "Good" },
//     moderate: { overall_aqi: 75, category: "Moderate" },
//     unhealthy: { overall_aqi: 165, category: "Unhealthy" }
// };

// // 2. The function the console is looking for
// function testAQI(level) {
//     const data = dummyAQIData[level];
//     if (data) {
//         document.getElementById("aqiValue").innerText = data.overall_aqi;
//         document.getElementById("aqiCategory").innerText = data.category;
        
//         // Remove old classes and add new one
//         document.body.classList.remove('good', 'moderate', 'unhealthy');
//         document.body.classList.add(level);
        
//         console.log("Background changed to:", level);
//     } else {
//         console.error("Use: testAQI('good'), testAQI('moderate'), or testAQI('unhealthy')");
//     }
// }
// // -------------------- INITIALIZATION --------------------
// window.onload = () => {
//     updateDateTime();
//     getUserLocation();
// };

// function updateDateTime() {
//     document.getElementById("dateTime").innerText = new Date().toLocaleString();
// }

// // -------------------- CORE FUNCTIONS --------------------
// function getUserLocation() {
//     if (navigator.geolocation) {
//         navigator.geolocation.getCurrentPosition(
//             position => {
//                 document.getElementById("locationText").innerText = "Current Location Detected";
//                 getAQI(); 
//             },
//             () => {
//                 console.log("GPS Denied. Defaulting to Ahmedabad.");
//                 document.getElementById("locationText").innerText = "Ahmedabad (Default)";
//                 getAQI(); 
//             }
//         );
//     }
// }

// async function getAQI() {
//     const location = document.getElementById("manualLocation").value || "Ahmedabad";
//     try {
//         const response = await fetch(`http://localhost:5000/get_aqi?city=${location}`);
//         if (!response.ok) throw new Error("Server error");
//         const data = await response.json();

//         document.getElementById("aqiValue").innerText = data.overall_aqi;
//         document.getElementById("aqiCategory").innerText = data.category;
        
//         updateBackground(data.overall_aqi);
//         showAQITable(data.predictions);

//         // Save for mapScript.js
//         localStorage.setItem("regionsData", JSON.stringify(data.regions));
//     } catch (error) {
//         alert("Backend server not reached. Check if it is running on port 5000.");
//     }
// }

// function renderAQIData(data) {
//     document.getElementById("aqiValue").innerText = data.overall_aqi;
//     document.getElementById("aqiCategory").innerText = data.category;
    
//     updateBackground(data.overall_aqi);
//     showAQITable(data.predictions);
// }

// function updateBackground(aqi) {
//     let bgClass = "good";
//     if (aqi > 50 && aqi <= 100) bgClass = "moderate";
//     if (aqi > 100) bgClass = "unhealthy";
    
//     document.body.className = bgClass; // Applies background from style.css
// }

// function showAQITable(predictions) {
//     const table = document.getElementById("aqiTable");
//     table.style.display = "table";
//     table.innerHTML = `<tr><th>Time</th><th>AQI</th></tr>` + 
//         predictions.map(p => `<tr><td>${p.time}</td><td>${p.aqi}</td></tr>`).join("");
// }

// function goToMap() {
//     window.location.href = "map.html";
// }
// // New function to render the 7-day data
// function showForecastTable(forecast) {
//     const table = document.getElementById("forecastTable");
//     table.innerHTML = `<tr><th>Date</th><th>Location</th><th>Predicted AQI</th></tr>`;
    
//     forecast.forEach(day => {
//         const row = table.insertRow();
//         row.innerHTML = `
//             <td>${day.date}</td>
//             <td>${day.location}</td>
//             <td class="${getAQIClass(day.aqi)}">${day.aqi}</td>
//         `;
//     });
// }
// function getUserLocation() {
//     if (navigator.geolocation) {
//         navigator.geolocation.getCurrentPosition(
//             position => {
//                 // Success: In a real app, you'd reverse-geocode coordinates here
//                 document.getElementById("locationText").innerText = "Location Detected";
//                 getAQI(); 
//             },
//             () => {
//                 // Failure: Instead of an alert, set a default city
//                 console.log("Location access denied. Using default.");
//                 document.getElementById("locationText").innerText = "Ahmedabad (Default)";
//                 getAQI(); 
//             }
//         );
//     }
// }
// -------------------- INITIALIZATION --------------------
window.onload = () => {
    updateDateTime();
    // Default dummy load for testing
    renderDummyData();
};

function updateDateTime() {
    document.getElementById("dateTime").innerText = new Date().toLocaleString();
}

// -------------------- DUMMY DATA FOR TESTING --------------------
function renderDummyData() {
    const data = {
        overall_aqi: 119,
        category: "Poor",
        predictions: [
            { time: "07:00 PM", aqi: 100 },
            { time: "09:00 PM", aqi: 102 },
            { time: "11:00 PM", aqi: 105 },
            { time: "01:00 AM", aqi: 110 }
        ]
    };

    document.getElementById("aqiValue").innerText = data.overall_aqi;
    document.getElementById("aqiCategory").innerText = data.category;
    document.getElementById("locationText").innerText = "Ahmedabad";
    
    updateBackground(data.overall_aqi);
    showAQITable(data.predictions);
}

// -------------------- CORE FUNCTIONS --------------------
async function getAQI() {
    const location = document.getElementById("manualLocation").value || "Ahmedabad";
    // For Frontend testing, we use dummy logic if fetch fails
    try {
        const response = await fetch(`http://localhost:5000/get_aqi?city=${location}`);
        const data = await response.json();
        renderAQIData(data, location);
    } catch (error) {
        console.warn("Backend not found, staying in Dummy/Test mode.");
        renderDummyData();
    }
}

function updateBackground(aqi) {
    let bgClass = "good";
    if (aqi > 50 && aqi <= 100) bgClass = "moderate";
    if (aqi > 100) bgClass = "unhealthy";
    document.body.className = bgClass;
}

function showAQITable(predictions) {
    const table = document.getElementById("aqiTable");
    table.style.display = "table";
    table.innerHTML = `<tr><th>Time</th><th>Predicted AQI</th></tr>` + 
        predictions.map(p => `<tr><td>${p.time}</td><td>${p.aqi}</td></tr>`).join("");
}

function goToMap() {
    // Redirects to our new charts page instead of the map
    window.location.href = "forecast.html";
}