const menuBtn = document.getElementById('menu-btn');
const menu = document.getElementById('menu');
menuBtn.addEventListener('click', () => {
menu.classList.toggle('hidden');
});

// After 3 seconds, fade out the flash message
setTimeout(function() {
	const flashContainer = document.getElementById('flash-container');
    if (flashContainer) {
        flashContainer.style.transition = "opacity 0.5s ease";
        flashContainer.style.opacity = "0";
        setTimeout(() => flashContainer.remove(), 500); // remove from DOM after fade
    }
}, 3000); // 3 seconds