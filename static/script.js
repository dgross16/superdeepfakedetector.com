// Load OpenCV into window
const OPENCV_URL = './static/opencv.js';
const loadOpenCv = function() {
    let script = document.createElement('script');
    script.setAttribute('async', '');
    script.setAttribute('type', 'text/javascript');
    script.addEventListener('load', async () => {
        if (cv.getBuildInformation) {
            console.log(cv.getBuildInformation());
            console.log("normal load");
        }
        else {
            // WASM
            if (cv instanceof Promise) {
                cv = await cv;
                // console.log(cv.getBuildInformation());
                console.log("OpenCV.js Loaded (wasm promise load)");
            } else {
                cv['onRuntimeInitialized']=()=>{
                    console.log(cv.getBuildInformation());
                    console.log("wasm other load");
                }
            }
        }
    });
    script.addEventListener('error', () => {
        console.log('Failed to load ' + OPENCV_URL);
    });
    script.src = OPENCV_URL;
    let node = document.getElementsByTagName('script')[1];
    node.parentNode.insertBefore(script, node);
};




window.onload = document.getElementById("input").addEventListener("change", function() {
    const results = document.getElementById('results');
    const media = URL.createObjectURL(this.files[0]);
    results.style.display = "none";
    setVideo(media);
    framerun();
});


const setVideo = (videoURL) => {
    const results = document.getElementById('results');
    const videoContainer = document.getElementById('videoContainer');
    const video = document.getElementById("video"); // gets the video file
    const droparea = document.getElementById('droparea');
    droparea.firstElementChild.style.display = "none"
    droparea.style.borderStyle = "none";
    video.src = videoURL; // sets the src of the video tag to the input video
    console.log(video.src)
    //droparea.style.borderStyle = "none";
    video.style.display = "block";
    videoContainer.style.display = "flex";
    results.style.display = "none";

};

const tryExample = function () {
    const examples = ['example1.mp4', 'example2.mp4', 'example3.mp4']
    const example = examples.splice(Math.floor(Math.random() * 3), 1);
    console.log(example);
    const videoURL = `examples/${example}`; // sets the src of the video tag to the input video
    console.log(videoURL);
    setVideo(videoURL);
    framerun();
}

const dnd = function () {
    const droparea = document.getElementById('droparea')

    const active = () => droparea.classList.add("green-border");
    const inactive = () => droparea.classList.remove("green-border");

    const prevents = (e) => e.preventDefault();
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evtName => {
        droparea.addEventListener(evtName, prevents);
    });

    ['dragenter', 'dragover'].forEach(evtName => {
        droparea.addEventListener(evtName, active);
    });

    ['dragleave', 'drop'].forEach(evtName => {
        droparea.addEventListener(evtName, inactive);
    });

    droparea.addEventListener('drop', handleDrop);

};

const handleDrop = function (e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    const media = URL.createObjectURL(files[0]);
    setVideo(media);
};

document.addEventListener("DOMContentLoaded", dnd);


const framerun = async function () {
    // Tensorflow.js model load
    const incep_resnetURL = 'models/incep_resnet/graph/model.json';
    //const xceptionURL = 'conversion/cnn_test/xception-2/graph/model.json';
    const model = await tf.loadGraphModel(incep_resnetURL);
    // Get Video Dimensions
    const video = document.getElementById('video');
    const vidHeight = video.videoHeight;
    const vidWidth = video.videoWidth;
    video.height = vidHeight;
    video.width = vidWidth;
    console.log(video.height + ' ' + video.width);

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let cap = new cv.VideoCapture(video);
    const FPS = 30;

    let duration = video.duration;

    let length = Math.floor(duration * FPS);

    video.muted = true;
    video.play();

    const faceModel = faceDetection.SupportedModels.MediaPipeFaceDetector;
    const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection',
    };
    const faceDetector = await faceDetection.createDetector(faceModel, detectorConfig);
    let count = 0;
    let all_preds = [];
    const results = document.getElementById('results');

    async function processVideo() {
        console.log(!video.paused);
        if (!video.paused){

        try {
            // start processing.
            const begin = Date.now();

            cap.read(src);
            cv.imshow('canvasOutput', src);
            const input = document.getElementById('canvasOutput');
            const image = tf.browser.fromPixels(input);


            //console.log(image.shape);
            const faces = await faceDetector.estimateFaces(image).then(faces => {return faces});
            //console.log(faces);

            if (faces == false) {
                console.log('no faces!');
            } else {
                // Face box and crop
                const width = faces[0]['box']['width']
                const height = faces[0]['box']['height']
                const padding = Math.min(width, height)/3
                //const padding = 0;

                console.log(padding)
                const x1 = faces[0]['box']['xMin'] - padding
                const x2 = faces[0]['box']['xMax'] + padding
                const y1 = faces[0]['box']['yMin'] - padding
                const y2 = faces[0]['box']['yMax'] + padding



                const box = tf.tensor1d([y1/vidHeight, x1/vidWidth, y2/vidHeight, x2/vidWidth]);



                //const croppedFace = tf.slice(image,)
                const croppedFace= tf.tidy(() => {
                    let x = tf.cast(tf.image.cropAndResize(tf.expandDims(image), tf.expandDims(box), [0], [256, 256]), 'int32');
                    x = tf.cast(x, 'float32');
                    // ([0, 255] - 127.5) / 127.5 = [-127.5, 127.5] / 127.5 = [-1, 1]
                    x = tf.cast(x, 'float32')
                    x = tf.div(x, tf.scalar(127.5))
                    x = tf.sub(x, tf.scalar(1.0))
                    return x;
                });
                /*
                const imageTensor = tf.tidy(() => {
                    let x = tf.image.cropAndResize(tf.expandDims(image), tf.expandDims(box), [0], [256, 256]);
                    // ([0, 255] - 127.5) / 127.5 = [-127.5, 127.5] / 127.5 = [-1, 1]
                    x = tf.cast(tf.div(x, tf.scalar(255)), 'float32');
                    return x;
                });

                const overlay = document.getElementById('faceCanvas');
                const drawn_img = await tf.browser.toPixels(tf.squeeze(imageTensor));
                const overlay_img = new ImageData(drawn_img, 256, 256);
                overlay.getContext("2d").putImageData(overlay_img, 0, 0);
                */

                console.log('after crop: ' + (Date.now() - begin).toString() + ' ms');


                //console.log(croppedFace);



                const inf = Date.now();

                const preds = await model.execute(croppedFace);

                console.log('inference: ' + (Date.now() - inf).toString() + ' ms');
                const prob = preds.arraySync()[0][0];
                console.log((Math.round(prob * 10000) / 100).toFixed(2))
                all_preds.push(Number((Math.round(prob * 10000) / 100).toFixed(2)));
                if (all_preds.length > 5) {
                    results.style.display = "block";
                    const real_avg = (all_preds.reduce((prev, curr) => prev + curr) / all_preds.length).toPrecision(2);
                    console.log(`real_avg: ${real_avg}`);

                    if (real_avg > 50) {
                        results.style.backgroundColor = "#5B841E"
                        results.style.borderColor = "#5B841E"
                        results.innerText = `${real_avg}% REAL`;
                    }
                    else {
                        results.style.backgroundColor = "#D90F0F"
                        results.style.borderColor = "#D90F0F"
                        results.innerText = `${100 - real_avg}% FAKE`;
                    }

                };


                tf.dispose([box, croppedFace]);

            };
            console.log('total: ' + (Date.now() - begin).toString() + ' ms');
            const delay = 1000/FPS - (Date.now() - begin);
            tf.dispose(image);
            if (video.readyState < 3) {
                return;
            };
            setTimeout(processVideo, delay);

        } catch (err) {
            console.log(err);
        };
        };

    };

    // schedule the first one.
    setTimeout(processVideo, 0);
};
