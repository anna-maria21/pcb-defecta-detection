const ctx5 = document.getElementById('marksChart').getContext('2d');


new Chart(ctx5, {
    type: 'bar',
    data: {
        labels: marksModels,
        datasets: [
            {
                label: 'Оцінка моделей користувачами',
                data: marks,
                backgroundColor: [
                    'rgba(255,181,102,0.7)',
                    'rgba(145,137,255,0.7)',
                    'rgba(255,137,253,0.7)',
                    'rgba(247,255,133,0.7)'
                ],
                borderColor: [
                    'rgba(255,181,102,0.7)',
                    'rgba(145,137,255,0.7)',
                    'rgba(255,137,253,0.7)',
                    'rgba(247,255,133,0.7)'
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
                text: 'Оцінка моделей користувачами',
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
                    text: 'Оцінка',
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
                    text: 'Пари моделей',
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