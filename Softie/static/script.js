function getDevices() {
    fetch('/devices')
    .then(response => response.json())
    .then(data => {
        const select =document.getElementById('camera');
        select.innerHTML = '';
        data.forEach(device => {
            select.innerHTML += `<option value="${device.id}">${device.name}</option>`;
        });
    });
}

function startStream() {
    const source = document.getElementById("camera").value;
    const fps = document.getElementById("fps").value;
    const blur = document.getElementById("blur").value;
    const background = document.getElementById("background").value;

    fetch(`/start?source=${source}&fps=${fps}&blur=${blur}&background=${background}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById("status").innerHTML = data.message;
    })
}

function stopStream() {
    fetch('/stop')
    .then(response => response.json())
    .then(data => {
        document.getElementById("status").innerHTML = data.message;
    })
}

function updateBlurValue(value) {
    document.getElementById('blurValue').innerText = `Blur Strength: ${value}`;
}