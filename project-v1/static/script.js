// static/script.js
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

canvas.width = window.innerWidth;  // Set canvas width to window width
canvas.height = window.innerHeight; // Set canvas height to window height

// Function to draw on the canvas
function draw(x, y) {
    if (drawing) {
        ctx.lineTo(x, y);
        ctx.strokeStyle = 'purple';
        ctx.lineWidth = 10;
        ctx.stroke();
    }
}

// Start drawing when the index finger is detected
const eventSource = new EventSource('/video_feed');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.x !== undefined && data.y !== undefined) {
        if (!drawing) {
            drawing = true;
            ctx.beginPath();
            ctx.moveTo(data.x, data.y);
        }
        draw(data.x, data.y);
    } else {
        drawing = false; // Stop drawing if no finger is detected
    }
};

canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.closePath();
});

canvas.addEventListener('mouseout', () => {
    drawing = false;
    ctx.closePath();
});
