document.getElementById('translate-btn').addEventListener('click', async () => {
    const inputText = document.getElementById('english-input').value; // Get the input text

    try {
        const response = await fetch('http://127.0.0.1:5000', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ word:inputText }) 
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        document.getElementById('santhali-output').value = data.santhali_word; 
    } 
    catch (error) {
        console.error('Error:', error);
    }
});
