// Access the video element
var videoElement = document.getElementById('camera');

// Initialize media stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        videoElement.srcObject = stream;
    })
    .catch(function (error) {
        console.error('Error accessing webcam:', error);
    });

// Record video and prepare for submission
var recordedChunks = [];
var mediaRecorder;

// Start recording when the "Start Recording" button is clicked
document.getElementById('startRecord').addEventListener('click', function () {
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(videoElement.srcObject);

    mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.start();
    document.getElementById('startRecord').disabled = true;
    document.getElementById('stopRecord').disabled = false;
});

// Stop recording when the "Stop Recording" button is clicked
document.getElementById('stopRecord').addEventListener('click', function () {
    mediaRecorder.stop();
    document.getElementById('startRecord').disabled = false;
    document.getElementById('stopRecord').disabled = true;
});

// Prepare and submit the recorded video when the form is submitted
document.getElementById('videoForm').addEventListener('submit', function (event) {
    event.preventDefault();

    var blob = new Blob(recordedChunks, { type: 'video/webm' });
    var reader = new FileReader();
    reader.onload = function () {
        var videoData = reader.result.split(',')[1];
        document.getElementById('videoData').value = videoData;
        document.getElementById('videoForm').submit();
    };
    reader.readAsDataURL(blob);
});
