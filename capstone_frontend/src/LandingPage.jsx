import { FaFileUpload, FaRobot, FaDownload } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import "./LandingPage.css";

const steps = [
  {
    icon: <FaFileUpload />,
    title: "Upload Document",
    desc: "Upload PDF or DOCX legal files securely.",
  },
  {
    icon: <FaRobot />,
    title: "Smart Processing",
    desc: "Extract clauses, entities, and key legal information automatically.",
  },
  {
    icon: <FaDownload />,
    title: "View Results",
    desc: "Get structured insights and simplified summaries instantly.",
  },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-container">

      <div className="hero">
        <h1 className="logo">LegalEase</h1>
      </div>

      <div className="steps-wrapper">
        <div className="steps-grid">
          {steps.map((step, index) => (
            <div className="step-card" key={index}>
              <div className="icon">{step.icon}</div>
              <h3>{step.title}</h3>
              <p>{step.desc}</p>
            </div>
          ))}
        </div>
        <button className="try-now-btn" onClick={() => navigate("/dashboard")}>
          Try Now
        </button>
      </div>
    </div>
  );
}
