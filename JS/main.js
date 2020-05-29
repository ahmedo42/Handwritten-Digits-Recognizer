var predictBtn = document.getElementById('predict');
var clearBtn = document.getElementById('clear');
var grid = document.getElementById('drawing');
var field = document.getElementById('result');
var paint = false;
var coordinates = {x:0 , y:0};
var model;

const ctx = grid.getContext("2d")
ctx.lineWidth = 10;
ctx.lineCap = "round";
ctx.strokeStyle = 'white';

window.addEventListener('load',async function loadModel(){
    model = undefined;
    model = await tf.loadLayersModel("models/model.json")
});

clearBtn.addEventListener("click",function clearCanvas(){
    ctx.clearRect(0, 0, grid.width, grid.height);
    field.innerHTML = "Draw and Predict a digit!"
});

predictBtn.addEventListener("click",async function makeClassification(){
    let tensor = preProcess(grid);
    let probs = await model.predict(tensor).data();
    let result = Array.from(probs);
    display(result);
});
grid.addEventListener('mousedown',function startPainting(e){
  paint = true;
  updatePosition(coordinates);
});

grid.addEventListener('mouseup',function stopPainting(){
    paint = false;
});
grid.addEventListener('mousemove',function sketch(e){
    if(paint){
        ctx.beginPath();
        ctx.moveTo(coordinates.x,coordinates.y);
        updatePosition(e);
        ctx.lineTo(coordinates.x,coordinates.y);
        ctx.stroke();
    }
});

function updatePosition(point){
    coordinates.x = point.clientX - grid.offsetLeft;
    coordinates.y = point.clientY - grid.offsetTop;
}

function preProcess(image){
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat();
    return tensor.div(255.0);
}


function display(result){
    var Label = result[0];
    var probability = 0;
 
    for (var i = 0; i < result.length; i++) {
        if (result[i] > probability) {
            Label = i;
            probability = result[i];
        }
    }
    field.innerHTML = "Predicted " + Label + " with " + Math.trunc( probability * 100 ) + "% Confidence";
}