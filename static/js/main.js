// import './style.css'
// import javascriptLogo from './javascript.svg'
// import viteLogo from '/vite.svg'
// import { setupCounter } from './counter.js'

// document.querySelector('#app').innerHTML = `
//   <div>
//     <a href="https://vite.dev" target="_blank">
//       <img src="${viteLogo}" class="logo" alt="Vite logo" />
//     </a>
//     <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank">
//       <img src="${javascriptLogo}" class="logo vanilla" alt="JavaScript logo" />
//     </a>
//     <h1>Hello Vite!</h1>
//     <div class="card">
//       <button id="counter" type="button"></button>
//     </div>
//     <p class="read-the-docs">
//       Click on the Vite logo to learn mori
//     </p>
//   </div>
// `

// setupCounter(document.querySelector('#counter'))

// REQUIRED_FEATURES = [
//   'age', 'gender', 'department', 'academic_year', 'cgpa', 'scholarship',
//   'nervous', 'worry', 'relax', 'annoyed', 'overthinking', 'restless', 'afraid'
// ]

// fetch("http://localhost:5000/predict", {
//   method: "POST",
//   headers: {
//     "Content-Type": "application/json"
//   },
//   body: JSON.stringify({
//     age: age,
//     gender: gender,
//     departement: departement,
//     academic_year:academic_year,
//     cgpa:cgpa,
//     scholarship:scholarship,
//     nervous:nervous,
//     worry:worry,
//     relax:relax,
//     annoyed:annoyed,
//     overthinking:overthinking,
//     restless:restless,
//     afraid:afraid
//   })
// })
// .then(res => res.json())
// .then(data => {
//   console.log(data.prediction);
// });

// fetch("http://127.0.0.1:5000/predict", {
//   method: "POST",
//   headers: {
//       "Content-Type": "application/json"
//   },
//   body: JSON.stringify(formData)  // Pastikan data sudah berupa JSON
// })

// console.log("Form data:", JSON.stringify(formData));



// document.getElementById('formData').addEventListener('submit', async (e) => {
//   e.preventDefault(); // Cegah form submit default

//   const inputText = document.getElementById('inputText').value;

//   const response = await fetch('/predict', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ input: inputText })
//   });

//   const data = await response.json();
//   document.getElementById('result').innerText = data.prediction;
// });
