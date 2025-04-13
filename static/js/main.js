$(document).ready(function () {
    var predicted = [];
    var original = [];
    var tomm_pred, mse;
    var err = true;

    $('.loader').css("display", "none");
    $(".results-container").css("display", "none");

    // Handle enter key on input
    $("#comp").keypress(function (e) {
        if (e.which == 13) {
            $("#button1").click();
        }
    });

    // Handle popular ticker clicks
    $(".ticker").click(function(e) {
        e.preventDefault();
        $("#comp").val($(this).text());
        $("#button1").click();
    });

    // Main button click handler
    $("#button1").click(function () {
        // Validate input
        const ticker = $("#comp").val().trim();
        if (!ticker) {
            showNotification("Please enter a valid ticker symbol", "error");
            return;
        }

        // Show loader and hide results
        $('.loader').css("display", "flex");
        $(".results-container").css("display", "none");

        // Format ticker and make API call
        const formattedTicker = ticker.toUpperCase();
        const apiUrl = 'http://127.0.0.1:5000/predict/' + formattedTicker;
        
        $.ajax({
            url: apiUrl,
            type: "GET",
            success: function (result) {
                predicted = JSON.parse(result['predicted']);
                original = JSON.parse(result['original']);
                tomm_pred = JSON.parse(result['tommrw_prdctn']);
                mse = JSON.parse(result['mn_sqre_err']);
                err = true;
                displayResults(formattedTicker);
            },
            error: function (error) {
                console.log(error);
                err = false;
                $('.loader').css("display", "none");
                showNotification("Error fetching data for " + formattedTicker + ". Please try another ticker.", "error");
            }
        });
    });

    // Function to display the results
    function displayResults(ticker) {
        if (err) {
            // Calculate percentage change for display
            const lastOriginalValue = original[original.length - 1];
            const percentChange = ((tomm_pred - lastOriginalValue) / lastOriginalValue * 100).toFixed(2);
            const changeDirection = percentChange >= 0 ? 'positive' : 'negative';
            const changeIcon = percentChange >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';

            // Update detail cards
            const detailsCards = `
                <div class="detail-card">
                    <h3>${ticker} Prediction</h3>
                    <div class="value">$${parseFloat(tomm_pred).toFixed(2)}</div>
                    <div class="indicator ${changeDirection}">
                        <i class="fas ${changeIcon} mr-1"></i> ${Math.abs(percentChange)}%
                    </div>
                </div>
                <div class="detail-card">
                    <h3>Mean Square Error</h3>
                    <div class="value">${parseFloat(mse).toFixed(4)}</div>
                    <div class="indicator">
                        <span class="text-muted">Prediction Accuracy</span>
                    </div>
                </div>
                <div class="detail-card">
                    <h3>Current Price</h3>
                    <div class="value">$${parseFloat(lastOriginalValue).toFixed(2)}</div>
                    <div class="indicator">
                        <span class="text-muted">Last Recorded Value</span>
                    </div>
                </div>
            `;
            
            $(".details-cards").html(detailsCards);
            
            // Create the chart
            createChart();
            
            // Show results container
            $('.loader').css("display", "none");
            $(".results-container").css("display", "block");
            
            // Scroll to results
            $('html, body').animate({
                scrollTop: $(".results-container").offset().top - 100
            }, 500);
        }
    }

    // Function to create the chart
    function createChart() {
        const ctx = document.getElementById('myChart1').getContext('2d');
        
        // Check if chart already exists and destroy it
        if (window.predictionChart) {
            window.predictionChart.destroy();
        }
        
        // Generate labels for the chart
        const labels = Array.from({ length: original.length }, (_, i) => i);
        
        // Find min and max values to set appropriate y-axis scale
        const allValues = [...original, ...predicted];
        const minValue = Math.min(...allValues) * 0.95; // Add 5% padding below
        const maxValue = Math.max(...allValues) * 1.05; // Add 5% padding above
        
        // Create new chart
        window.predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction',
                    data: predicted,
                    fill: false,
                    backgroundColor: 'rgb(247, 33, 4)',
                    borderColor: 'rgb(247, 33, 4)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Actual Price',
                    data: original,
                    fill: false,
                    backgroundColor: 'rgb(26, 26, 239)',
                    borderColor: 'rgb(26, 26, 239)',
                    borderWidth: 2,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2.5, // This controls the height relative to width
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': $' + context.raw.toFixed(2);
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hitRadius: 10,
                        hoverRadius: 5
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: minValue, // Set minimum value
                        max: maxValue, // Set maximum value
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            },
                            maxTicksLimit: 8 // Limit the number of ticks on y-axis
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            autoSkip: true,
                            maxTicksLimit: 10
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    // Function to show notifications
    function showNotification(message, type) {
        // Check if notification exists and remove it
        if ($('.notification').length) {
            $('.notification').remove();
        }

        // Create notification element
        const notification = $('<div>').addClass('notification ' + type);
        const icon = type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle';
        
        notification.html(`
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        `);
        
        // Add to body
        $('body').append(notification);
        
        // Show notification
        setTimeout(() => {
            notification.addClass('show');
        }, 100);
        
        // Hide after 3 seconds
        setTimeout(() => {
            notification.removeClass('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    // Add notification styles
    const notificationStyles = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background-color: white;
            color: #1e293b;
            border-radius: 8px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            transform: translateX(120%);
            transition: transform 0.3s ease;
            z-index: 1000;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification i {
            margin-right: 0.75rem;
            font-size: 1.25rem;
        }
        
        .notification.error {
            border-left: 4px solid #ef4444;
        }
        
        .notification.error i {
            color: #ef4444;
        }
        
        .notification.success {
            border-left: 4px solid #22c55e;
        }
        
        .notification.success i {
            color: #22c55e;
        }
    `;
    
    $('head').append('<style>' + notificationStyles + '</style>');
});