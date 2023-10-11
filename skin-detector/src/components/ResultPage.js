import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './ResultPage.css';

const ResultPage = () => {
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    const imagePath = localStorage.getItem('uploadedImagePath');

    axios
      .post('http://localhost:5000/predict', { image_path: imagePath })
      .then((response) => {
        const predictedLabel = response.data.predicted_label;
        const precisionForPredictedLabel = response.data.precision_for_predicted_label;

        setPredictionResult({ predictedLabel, precisionForPredictedLabel });
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  }, []);

  return (
    <div>
      {predictionResult && (
        <div>
          <h2 className='title'>Prediction Label:</h2>
          <p>Lesion Name: {predictionResult.predictedLabel}</p>
        </div>
      )}
    </div>
  );
};

export default ResultPage;
