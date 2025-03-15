import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <Link to="/" className="logo">
          <span className="logo-mindset">MINDSET</span>
          <span className="logo-subtext">AI-Powered News Analytics</span>
        </Link>
        <nav className="nav-links">
          <Link to="/" className="nav-link">Home</Link>
          <Link to="/analyze" className="nav-link">Analyze Article</Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;