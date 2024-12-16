const ctx = document.getElementById('classificationBarChart').getContext('2d');

new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['VGG', 'ResNet'],
        datasets: [
            {
                label: 'Середній час класифікації (ms)',
                data: [avgClass[0]['total_time'], avgClass[1]['total_time']], // Середній час класифікації для моделей
                backgroundColor: [
                    'rgba(217,208,132,0.7)',
                    'rgba(54, 162, 235, 0.7)'
                ],
                borderColor: [
                    'rgba(217,208,132,0.7)',
                    'rgba(54, 162, 235, 1)'
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
                text: 'Середній час класифікації',
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
                    text: 'Моделі класифікації',
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