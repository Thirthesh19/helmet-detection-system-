document.addEventListener('DOMContentLoaded', () => {
    // --- File Input Styling ---
    const fileInput = document.querySelector('.file-input');
    const fileMsg = document.querySelector('.file-msg');

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileMsg.textContent = e.target.files[0].name;
            } else {
                fileMsg.textContent = 'or drag and drop here';
            }
        });
    }

    // --- Stats Fetching (Dashboard only) ---
    // Check if we are on the dashboard by looking for stat elements
    const totalRiders = document.getElementById('total_riders');
    const helmetUsers = document.getElementById('helmet_users');
    const noHelmetUsers = document.getElementById('no_helmet_users');
    const totalViolations = document.getElementById('total_violations');

    if (totalRiders) {
        // Fetch stats every 2 seconds to update dashboard in real-time
        setInterval(fetchStats, 2000);
        // Initial fetch
        fetchStats();
    }

    function fetchStats() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                updateStat(totalRiders, data.total_riders);
                updateStat(helmetUsers, data.helmet_users);
                updateStat(noHelmetUsers, data.no_helmet_users);
                updateStat(totalViolations, data.total_violations);
            })
            .catch(err => console.error("Error fetching stats:", err));
    }

    function updateStat(element, newValue) {
        const currentValue = parseInt(element.innerText) || 0;
        if (currentValue !== newValue) {
            animateValue(element, currentValue, newValue, 500);
            // Re-trigger CSS animation
            element.classList.remove('pop-anim');
            void element.offsetWidth; // trigger reflow
            element.classList.add('pop-anim');
        }
    }

    // Smooth counter animation
    function animateValue(obj, start, end, duration) {
        if (start === end) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
