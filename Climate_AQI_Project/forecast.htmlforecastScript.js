// Dummy Data for Testing
const dummyData = {
    sevenDay: {
        labels: ['Dec 30', 'Dec 31', 'Jan 1', 'Jan 2', 'Jan 3', 'Jan 4', 'Jan 5'],
        values: [119, 103, 105, 130, 147, 95, 88],
        categories: ['Poor', 'Poor', 'Poor', 'Poor', 'Poor', 'Moderate', 'Moderate']
    },
    hourly: {
        labels: ['7 PM', '9 PM', '11 PM', '1 AM', '3 AM', '5 AM', '7 AM', '9 AM', '11 AM', '1 PM', '3 PM', '5 PM'],
        values: [100, 102, 105, 110, 108, 98, 95, 115, 125, 140, 135, 120]
    }
};

// Common Chart Options for White Labels
const commonOptions = {
    responsive: true,
    plugins: {
        legend: { labels: { color: 'white', font: { size: 14 } } }
    },
    scales: {
        y: {
            beginAtZero: true,
            title: { display: true, text: 'AQI Value', color: 'white', font: { size: 16 } },
            ticks: { color: 'white' },
            grid: { color: 'rgba(255,255,255,0.1)' }
        },
        x: {
            title: { display: true, text: 'Time Period', color: 'white', font: { size: 16 } },
            ticks: { color: 'white' },
            grid: { display: false }
        }
    }
};

// 1. Initialize 7-Day Bar Chart
const ctx7 = document.getElementById('sevenDayChart').getContext('2d');
new Chart(ctx7, {
    type: 'bar',
    data: {
        labels: dummyData.sevenDay.labels,
        datasets: [{
            label: 'Daily Predicted AQI',
            data: dummyData.sevenDay.values,
            backgroundColor: dummyData.sevenDay.values.map(v => v > 100 ? '#f97316' : '#fbbf24'),
            borderColor: 'white',
            borderWidth: 1,
            borderRadius: 5
        }]
    },
    options: commonOptions
});

// 2. Initialize Hourly Line Chart
const ctxH = document.getElementById('hourlyChart').getContext('2d');
new Chart(ctxH, {
    type: 'line',
    data: {
        labels: dummyData.hourly.labels,
        datasets: [{
            label: 'Hourly Trend',
            data: dummyData.hourly.values,
            borderColor: '#0ea5e9',
            backgroundColor: 'rgba(14, 165, 233, 0.3)',
            fill: true,
            tension: 0.4,
            pointRadius: 5,
            pointBackgroundColor: 'white'
        }]
    },
    options: commonOptions
});
