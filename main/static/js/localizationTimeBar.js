const ctx1 = document.getElementById('localizationBarChart').getContext('2d');

new Chart(ctx1, {
    type: 'bar',
    data: {
        labels: ['Yolo v8', 'Faster R-CNN'],
        datasets: [
            {
                label: 'Середній час локалізації (ms)',
                data: [avgLoc[0]['total_time'], avgLoc[1]['total_time']],
                backgroundColor: [
                    'rgba(160,227,113,0.7)',
                    'rgba(225,96,118,0.7)'
                ],
                borderColor: [
                    'rgba(160,227,113,0.7)',
                    'rgba(225,96,118,0.7)'
                ],
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: false,
        plugins: {
            legend: {
                display: false,
                labels: {
                    color: 'white',
                    font: {
                        size: 16
                    }
                }
            },
            title: {
                display: true,
                text: 'Середній час локалізації',
                color: 'white',
                font: {
                    size: 24
                },
                padding: {
                    bottom: 40
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Час (ms)',
                    color: 'white',
                    font: {
                        size: 18
                    }
                },
                ticks: {
                    color: 'white',
                    font: {
                        size: 16
                    }
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Моделі локалізації',
                    color: 'white',
                    font: {
                        size: 18
                    }
                },
                ticks: {
                    color: 'white',
                    font: {
                        size: 16
                    }
                }
            }
        }
    }
});