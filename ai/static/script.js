document.getElementById('textForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const input = document.getElementById('userInput').value;
  const selectedFile = document.getElementById('fileSelect').value;

  try {
    const response = await fetch('/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: input, selectedFile: selectedFile })
    });

    const result = await response.json();
    console.log(result);
    document.getElementById('response').textContent = `Server says: ${result.message}`;
  } catch (err) {
    console.error(err);
    document.getElementById('response').textContent = 'Error contacting server.';
  }
});