const ctx2 = document.getElementById('classificationDonatChart').getContext('2d');

new Chart(ctx2, {
    type: 'pie',
    data: {
        labels: ['VGG', 'ResNet'],
        datasets: [
            {
                labels: 'Порівняння використання моделей класифікації',
                data: [countClass[0]['count_class'], countClass[1]['count_class']],
                backgroundColor: [
                    'rgba(198,162,243,0.7)',
                    'rgba(155,243,188,0.7)'
                ],
                borderColor: [
                    'rgba(198,162,243,0.7)',
                    'rgba(155,243,188,0.7)'
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
                text: 'Порівняння використання моделей класифікації',
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