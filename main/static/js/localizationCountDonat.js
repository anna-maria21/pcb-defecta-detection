const ctx3 = document.getElementById('localizationDonatChart').getContext('2d');

new Chart(ctx3, {
    type: 'pie',
    data: {
        labels: ['Yolo v8', 'Faster R-CNN'],
        datasets: [
            {
                labels: 'Порівняння використання моделей локалізації',
                data: [countLoc[0]['count_loc'], countLoc[1]['count_loc']],
                backgroundColor: [
                    'rgba(255,173,163,0.7)',
                    'rgba(122,208,199,0.7)'
                ],
                borderColor: [
                    'rgba(255,173,163,0.7)',
                    'rgba(122,208,199,0.7)'
                ],
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: false,
        plugins: {
            legend: {
               position: 'bottom',
               labels: {
                  color: 'white',
                  font: {
                      size: 18
                  }
              }
            },
            title: {
                display: true,
                text: 'Порівняння використання моделей локалізації',
                color: 'white',
                font: {
                    size: 24
                },
                padding: {
                    bottom: 40
                }
            }
        }
    }
});