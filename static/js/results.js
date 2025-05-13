
// File: results.js
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have chart canvases on this page
    const chartCanvases = [
        document.getElementById('precisionChart'),
        document.getElementById('recallChart'),
        document.getElementById('lossChart'),
        document.getElementById('mapChart')
    ];
    
    // Only initialize charts if they exist on the page
    if (chartCanvases.every(canvas => canvas)) {
        console.log('Initializing metric charts');
        // The charts are initialized directly in the results.html template
    }
    
    // Sync MLFlow Button functionality
    const syncMlflowBtn = document.getElementById('sync-mlflow-btn');
    if (syncMlflowBtn) {
        syncMlflowBtn.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            
            // Disable button and show loading state
            this.disabled = true;
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Syncing...';
            
            // Call API to sync MLFlow data
            fetch(`/api/job/${jobId}/sync_mlflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success message
                    alert('Metrics and artifacts synced with MLFlow successfully!');
                    // Reload page to show updated data
                    window.location.reload();
                } else {
                    // Show error message
                    alert('Error: ' + (data.error || 'Failed to sync with MLFlow'));
                    // Reset button
                    this.disabled = false;
                    this.innerHTML = originalText;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error syncing with MLFlow. See console for details.');
                // Reset button
                this.disabled = false;
                this.innerHTML = originalText;
            });
        });
    }
});
