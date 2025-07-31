document.getElementById('textForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const input = document.getElementById('userInput').value;
  const selectedFile = document.getElementById('fileSelect').value;
  const responseDiv = document.getElementById('response');
  const loaderDiv = document.getElementById('loader');

  // Hide response and show loader
  responseDiv.classList.remove('visible');
  responseDiv.textContent = ''; // Clear previous response
  loaderDiv.style.display = 'block';

  try {
    const response = await fetch('/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: input, selectedFile: selectedFile })
    });

    const result = await response.json();
    console.log(result);
    responseDiv.textContent = result.message || 'No response from server.';
    responseDiv.classList.add('visible'); // Show response
  } catch (err) {
    console.error(err);
    responseDiv.textContent = 'Error contacting server.';
    responseDiv.classList.add('visible'); // Show response even on error
  } finally {
    loaderDiv.style.display = 'none'; // Hide loader
  }
});