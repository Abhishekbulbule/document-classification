import React, { useState } from "react";
import axios from "axios";

function App() {
  const categories = [
    "Atheism Discussions",
    "Computer Graphics",
    "Windows OS Miscellaneous",
    "IBM PC Hardware",
    "Mac Hardware",
    "X Window System",
    "For Sale Items",
    "Automobiles",
    "Motorcycles",
    "Baseball",
    "Hockey",
    "Cryptography",
    "Electronics",
    "Medical Science",
    "Space Science",
    "Christian Religion",
    "Gun Politics",
    "Middle East Politics",
    "Miscellaneous Politics",
    "Religion Miscellaneous"
]

  const [document, setDocument] = useState("");
  const [category, setCategory] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:8000/classify/", {
        document,
      });
      setCategory(categories[response.data.category]);
    } catch (error) {
      console.error("Error classifying document:", error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-image justify-center items-center">
      <div className="h-[20%] w-[60%] py-3 mt-10">
        <span className="self-center text-xl font-bold sm:text-2xl text-primary italic">
          Classify
        </span>

      </div>
      <div className="h-[80%] w-[30%] flex justify-start items-center  flex-col gap-5 ">
        <h1 className="text-2xl font-semibold">Classify content here</h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 w-[100%]">
          <textarea
            className="p-2 border rounded-md  focus:border-l-blue-500"
            value={document}
            onChange={(e) => setDocument(e.target.value)}
            placeholder="Enter document text"
          />
          <button className="bg-purple-500 p-3 rounded-lg text-white font-medium" type="submit">Classify</button>
        </form>
        {category && <p className="text-4xl font-bold">Category: {category}</p>}
      </div>
    </div>
  );
}

export default App;
