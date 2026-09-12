const form = document.getElementById("churnForm");

const result = document.getElementById("result");
const loading = document.getElementById("loading");

const predictionElement = document.getElementById("prediction");
const probabilityElement = document.getElementById("probability");
const riskElement = document.getElementById("risk");

const predictButton = document.getElementById("predictButton");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    predictButton.disabled = true;
    loading.classList.remove("hidden");
    result.classList.add("hidden");


    const customerData = {

        gender: document.getElementById("gender").value,

        SeniorCitizen: Number(
            document.getElementById("SeniorCitizen").value
        ),

        Partner: document.getElementById("Partner").value,

        Dependents: document.getElementById("Dependents").value,

        tenure: Number(
            document.getElementById("tenure").value
        ),

        PhoneService: document.getElementById("PhoneService").value,

        MultipleLines: document.getElementById("MultipleLines").value,

        InternetService: document.getElementById("InternetService").value,

        OnlineSecurity: document.getElementById("OnlineSecurity").value,

        OnlineBackup: document.getElementById("OnlineBackup").value,

        DeviceProtection: document.getElementById("DeviceProtection").value,

        TechSupport: document.getElementById("TechSupport").value,

        StreamingTV: document.getElementById("StreamingTV").value,

        StreamingMovies: document.getElementById("StreamingMovies").value,

        Contract: document.getElementById("Contract").value,

        PaperlessBilling: document.getElementById("PaperlessBilling").value,

        PaymentMethod: document.getElementById("PaymentMethod").value,

        MonthlyCharges: Number(
            document.getElementById("MonthlyCharges").value
        ),

        TotalCharges: Number(
            document.getElementById("TotalCharges").value
        )
    };


    try {

        const response = await fetch("/api/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)
        });


        if (!response.ok) {

            throw new Error(
                `Server error: ${response.status}`
            );

        }


        const data = await response.json();


        predictionElement.textContent = data.prediction;

        probabilityElement.textContent =
            `${data.churn_probability}%`;

        riskElement.textContent = data.risk;


        result.classList.remove("hidden");


    } catch (error) {

        alert(
            "Unable to get prediction. Please try again."
        );

        console.error(error);

    } finally {

        loading.classList.add("hidden");

        predictButton.disabled = false;

    }

});
