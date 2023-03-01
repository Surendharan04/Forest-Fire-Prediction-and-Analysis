function checkCredentials(event) {
    event.preventDefault(); // prevents the form from submitting
    let username = document.getElementById('user').value;
    let password = document.getElementById('pass').value;
    if (username === 'Forest' && password === 'pass') {
        window.location.href = "/templates/main.html";
    } else {
        alert('Invalid credentials. Please try again.');
    }
}