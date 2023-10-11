import React from 'react';
import './App.css';
import DropfileInput from './components/DropFileInput';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ResultPage from './components/ResultPage';

function App() {
    const onFileChange = (file) => {
        console.log(file);
    };

    return (
        <Router>
            <div className="box">
                <h2 className="header">Skin Detector</h2>
                <Routes>
                    <Route path="/" element={<DropfileInput onFileChange={onFileChange}/>} />
                    <Route path="/success" element={<ResultPage/>} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
