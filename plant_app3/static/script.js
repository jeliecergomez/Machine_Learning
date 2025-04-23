const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    });

captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, 224, 224);

    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(res => res.text())
        .then(html => {
            document.open();
            document.write(html);
            document.close();
        });
    }, 'image/jpeg');
});
