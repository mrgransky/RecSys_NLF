// Define global variables
let globalQuery = "{{ input_query|safe }}";
let digi_base_url = "https://digi.kansalliskirjasto.fi/search?query=";
let recommendationResults;
let myWindow;

// Search NLF function
function searchNLF() {
		var query = document.getElementsByName("query")[0].value.trim();
		if (query !== "") {
				var libLinkContainer = document.getElementById("libraryLinkContainer");

				// Create a link element
				var link = document.createElement("a");
				link.href = digi_base_url + encodeURIComponent(query);
				link.target = "_blank";
				link.textContent = "Click here to open National Library Results";

				// Clear any previous content
				libLinkContainer.innerHTML = "";

				// Append the link to the container
				libLinkContainer.appendChild(link);

				libLinkContainer.classList.add('blur-background');
		} else {
				alert('Enter a valid search query to proceed!');
		}
}

// Clear NLF search function
function clearNlfSearch() {
		// Clear the search results
		var libLinkContainer = document.getElementById("libraryLinkContainer");
		libLinkContainer.innerHTML = "";

		// Clear the search input field
		var queryInput = document.getElementsByName("query")[0];
		queryInput.value = "";
		queryInput.placeholder = "Hakusana | Sökord | Query prompt";

		// Remove blur effect from all elements with the class .blur-background
		var blurElements = document.querySelectorAll('.blur-background');
		blurElements.forEach(function(element) {
				element.classList.remove('blur-background');
		});
}

// Show loading spinner function
function showLoadingSpinner() {
		document.getElementById("loadingSpinner").style.display = "flex";
}

// Hide loading spinner function
function hideLoadingSpinner() {
		document.getElementById("loadingSpinner").style.display = "none";
}

// Recommend me function
function recommendMe(event) {
		var query = document.getElementsByName("query")[0].value.trim();
		document.getElementById("isRecSys").value = true; // Update the hidden input field
		showLoadingSpinner();
		if (query !== "") {
				document.getElementById("searchForm").submit();
		} else {
				alert('Oops! Enter a valid search query to proceed!');
				hideLoadingSpinner();
				event.preventDefault();
		}
}

// Clear RecSys function
function clearRecSys() {
		document.getElementsByName("query")[0].value = "";
		document.getElementsByName("query")[0].placeholder = "Hakusana | Sökord | Query prompt";
		document.getElementById("recSysIntroContainer").remove();
		document.getElementById("recSysResultsContainer").remove();
		document.getElementById("recSysSlider").remove();
		document.getElementById("sliderValueDisplay").remove();
		document.getElementById("recSysSliderLbl").remove();

		// Close the opened window if it exists and is not closed
		if (myWindow && !myWindow.closed) {
				myWindow.close();
		}

		// Translate the container back to its original position
		var container = document.querySelector('.container');
		container.style.transform = 'translateX(0)';
}

// Update slider value function
function updateSliderValue(sliderValue) {
		document.getElementById('recSysSlider').value = sliderValue;
		document.getElementById('sliderValueDisplay').textContent = sliderValue;
		updateResultsBasedOnSliderValue();
}

// Update results based on slider value function
function updateResultsBasedOnSliderValue() {
		var sliderValue = document.getElementById('recSysSlider').value;
		var recsysRes = document.getElementById('recSysResultsContainer');
		recsysRes.innerHTML = "";
		for (let i = 0; i < sliderValue; i++) {
				var recSysLink_i = digi_base_url + encodeURIComponent(globalQuery + " " + recommendationResults[i]);
				recsysRes.innerHTML += `<p><a class="rec-link" href='${recSysLink_i}' target='_blank'><span style="font-size: 75%;">${globalQuery}</span> + ${recommendationResults[i]}</a></p>`;
		}
		// Add event listener to all recommendation links
		var recLinks = document.querySelectorAll('.rec-link');
		recLinks.forEach(function(link) {
				link.addEventListener('click', function(event) {
						event.preventDefault(); // Prevent default link behavior

						// Smoothly shift the page layout to the left
						var container = document.querySelector('.container');
						container.style.transform = 'translateX(-30%)';

						// Get the URL of the clicked link
						var url = link.href;

						if (myWindow) {
								try {
										myWindow.close();
								} catch (error) {
										console.error("Error closing window:", error);
								}
						}

						// Calculate the width and height of the window
						const screenWidth = window.screen.width;
						const screenHeight = window.screen.height;
						const width = Math.round(screenWidth * 0.5); // 50% width
						const height = screenHeight;

						// Calculate the left position to place the window on the right side of the screen
						const left = screenWidth - width;

						// Open the new window
						myWindow = window.open(url, "", `width=${width}, height=${height}, left=${left}, top=0`);
						myWindow.focus();
				});
		});
}

// Event listener for DOMContentLoaded event
document.addEventListener("DOMContentLoaded", function() {
		// Update results based on slider value
		updateResultsBasedOnSliderValue();
});

// Event listener for search input field focus
document.addEventListener("DOMContentLoaded", function() {
		const searchInputField = document.querySelector('.search-input-field');
		const helpContainer = document.querySelector('.help-container');
		const foldElement = document.querySelector('.fold');

		searchInputField.addEventListener('focus', function() {
				helpContainer.style.display = 'flex'; // Show the help container
				foldElement.classList.remove('unfolder'); // Remove the unfold class to fold element
				foldElement.classList.add('fold'); // Ensure the initial folded state
		});

		// Toggle fold/unfold when clicking on the help container
		helpContainer.addEventListener('click', function(event) {
				// Toggle the fold class
				foldElement.classList.toggle('unfold');

				// Prevent event propagation to parent elements
				event.stopPropagation();
		});

		// Close the help container when clicking outside of it
		document.addEventListener('click', function(event) {
				if (!searchInputField.contains(event.target)) {
						helpContainer.style.display = 'none'; // Hide the help container
						foldElement.classList.remove('unfolder'); // Remove the unfold class
						foldElement.classList.add('fold'); // Ensure the folded state
				}
		});
});