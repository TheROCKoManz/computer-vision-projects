var socket = io.connect(
  window.location.protocol + "//" + document.domain + ":" + location.port
);
socket.on("connect", function () {
  console.log("Connected...!", socket.connected);
});

var canvasCamera = document.getElementById("canvasCamera");
var contextCamera = canvasCamera.getContext("2d");
const video = document.querySelector("#videoElement");

video.width = 600;
video.height = 400;

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
    })
    .then(function (stream) {
      video.srcObject = stream;
      video.play();
    })
    .catch(function (error) {
      console.error('Error accessing webcam:', error);
    });
}

const FPS = 10;
setInterval(() => {
  width = video.width;
  height = video.height;
  contextCamera.drawImage(video, 0, 0, width, height);
  var data = canvasCamera.toDataURL("image/jpeg", 0.5);
  contextCamera.clearRect(0, 0, width, height);
  socket.emit("image", data);
}, 1000 / FPS);

socket.on("processed_image", function (image) {
  console.log("Received Processed Image Data:", image);
  photo.setAttribute("src", image);
});
