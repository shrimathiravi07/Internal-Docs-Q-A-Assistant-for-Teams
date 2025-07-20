async function askQuestion() {
    const question = document.getElementById("questionInput").value;
  
    const res = await fetch("http://127.0.0.1:5000/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ question })
    });
  
    const data = await res.json();
    document.getElementById("answer").innerText = data.answer;
  }
  