import React, { useRef, useState } from 'react';
import { PropTypes } from 'prop-types';
import '../components/DropFile.css';
import uploadImage from '../assets/cloud-upload-regular-240.png';
import { ImageConfig } from '../config/ImageConfig';
import { useNavigate } from 'react-router-dom';

const DropfileInput = (props) => {
  const wrapperRef = useRef(null);
  const navigate = useNavigate();

  const [fileList, setfileList] = useState([]);
  const [checkBoxChecked, setCheckBoxChecked] = useState(false);

  const onDragEnter = () => wrapperRef.current.classList.add('dragover');
  const onDragLeave = () => wrapperRef.current.classList.remove('dragover');
  const onDrop = () => wrapperRef.current.classList.remove('dragover');

  const onFileDrop = (e) => {
    const newFile = e.target.files[0];
    if (newFile && checkBoxChecked && fileList.length === 0) {
      const updateList = [newFile];
      setfileList(updateList);
      props.onFileChange(updateList);

      const formData = new FormData();
      formData.append('image', newFile);

      const handleFileUpload = (formData) => {
        fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData,
        })
          .then((response) => response.blob())
          .then((data) => {
            console.log(data);
            localStorage.setItem('uploadedImagePath', `E:/Licenta/uploads/saved-images/${newFile.name}`);
            navigate('/success');
          })
          .catch((error) => {
            console.error(error);
          });
      };

      handleFileUpload(formData);
    }
  };

  const handleCheckBoxChange = (e) => {
    setCheckBoxChecked(e.target.checked);
  };

  return (
    <>
      <div
        ref={wrapperRef}
        className='drop-file-input'
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div className='drop-file-input__label'>
          <img src={uploadImage} alt='' />
          <p>Drag & Drop your image</p>
        </div>
        <input type='file' value='' onChange={onFileDrop} />
      </div>
      <div>
        <input
          type='checkbox'
          className='checkbox'
          checked={checkBoxChecked}
          onChange={handleCheckBoxChange}
        />
        <label className='checkbox-label'>
          I agree to the terms and conditions
        </label>
      </div>

      {fileList.length > 0 ? (
        <div className='drop-file-preview'>
          <p className='drop-file-preview__title'>
            Ready to upload.
          </p>
          {fileList.map((item, index) => (
            <div key={index} className='drop-file-preview__item'>
              <img src={ImageConfig['default']} alt='' />
              <div className='drop-file-preview__item__info'>
                <p>{item.name}</p>
                <p>{item.size}B</p>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </>
  );
};

DropfileInput.propTypes = {
  onFileChange: PropTypes.func,
};

export default DropfileInput;
