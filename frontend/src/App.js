import React, { useState } from "react";
import { Basic as Dropzone } from "./Components/DropZone.js";
import { FiExternalLink } from "react-icons/fi";
import { VscLoading } from "react-icons/vsc";
import { FishSpecies } from "./species.js"

const species = FishSpecies()

const App = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onResetHandler = () => {
    setImage(null);
    setResult(null);
  };
  const onPredictHandler = () => {
    // console.log(image);
    // TODO

    const url = "http://localhost:5000/predict";
    setLoading(true);
    fetch(url, {
      method: "POST",
      body: JSON.stringify(image.split(",")[1]),
      headers: new Headers({
        "Content-Type": "application/json",
      }),
    })
      .then((res) => res.json())
      .catch((error) => console.error("Error:", error))
      .then((response) => {
        console.log("Success:", response);
        setResult(
          response
        );
        setLoading(false);
      });
  };

  console.log(result);

  return (
    <div className="flex flex-col h-full items-center justify-center bg-blue-500 select-none">
      <div className="flex flex-col items-center bg-white w-96 rounded-lg shadow-xl py-5 px-8">
        <p className="uppercase text-center font-mono text-blue-500 text-3xl font-extrabold">
          Fish detector
        </p>
        <p className="mt-4 text-gray-700 text-justify text-sm">
          This is an APP that detects harvested fish in an image using deep learning model, YOLACT.
        </p>

        <div className="item-center mt-4 w-full">
          {!image && <Dropzone setImage={setImage} />}
          {image && !result && !loading && (
            <>
              <p className="text-sm uppercase font-medium mb-2">
                👇 your image
              </p>
              <img className="rounded-md w-full" src={image} />
            </>
          )}
          {loading && (
            <>
              <VscLoading className="my-4 mx-auto text-gray-500 text-3xl animate-spin" />
            </>
          )}
          {result && !loading && (
            <>
              <div className="flex-center">
                <div className="text-center w-12/12">
                  <img className="rounded-md w-full" src={`data:image/jpeg;base64,${result.image}`} />
                </div>
                <div className="flex flex-col mt-4 justify-between w-12/12">
                  {result["class"].map((cls, index) => (
                    <div className="flex justify-between">
                      <div className="w-full px-2 text-xl text-blue-500 rounded-md">
                        {index + 1}{". "}
                        {result["commonname"]}
                      </div>
                      <a
                        className="text-blue-500 my-auto"
                        target="_new"
                        href={result["link"]}
                      >
                        <FiExternalLink />
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        <div className="flex ml-auto mt-4 gap-2">
          <button
            type="button"
            onClick={onResetHandler}
            class="w-24 transition duration-200 text-md bg-gray-400 rounded-md p-1 items-center justify-center text-white hover:bg-gray-300 focus:outline-none"
          >
            Reset
          </button>
          {!result && !loading && (
            <button
              type="button"
              onClick={onPredictHandler}
              class="w-24 transition duration-200 text-md bg-yellow-400 rounded-md p-1 items-center justify-center text-white hover:bg-yellow-300 focus:outline-none"
            >
              Predict!
            </button>
          )}
        </div>
      </div>
      <span className="mt-2 text-sm text-blue-300">© 2021 Kuan-Ting and Hong-Yang</span>
    </div>
  );
};

export default App;
