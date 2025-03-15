// MINDSET Rust Acceleration Module
// This module provides optimized implementations of computationally 
// intensive operations for the MINDSET application.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::collections::HashSet;

/// Process text for feature extraction
/// 
/// This function performs preprocessing on text data to prepare it for feature extraction.
/// It handles tokenization, lowercasing, and punctuation handling in an optimized way.
/// 
/// Arguments:
///     text: The input text to process
///     lowercase: Whether to convert to lowercase (default: true)
///     remove_punctuation: Whether to remove punctuation (default: true)
/// 
/// Returns:
///     Processed text ready for feature extraction
#[pyfunction]
#[pyo3(signature = (text, lowercase=true, remove_punctuation=true))]
fn process_text(text: &str, lowercase: bool, remove_punctuation: bool) -> String {
    let mut result = String::with_capacity(text.len());
    
    if lowercase {
        result = text.to_lowercase();
    } else {
        result = text.to_string();
    }
    
    if remove_punctuation {
        result = result
            .chars()
            .filter(|c| !c.is_ascii_punctuation() || *c == '\'')
            .collect();
    }
    
    result
}

/// Extract n-grams from text
/// 
/// Efficiently extracts n-grams (contiguous sequences of n words) from text.
/// 
/// Arguments:
///     text: The input text
///     n: The size of n-grams to extract
/// 
/// Returns:
///     A vector of n-grams
#[pyfunction]
fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut ngrams = Vec::new();
    
    if words.len() < n {
        return ngrams;
    }
    
    for i in 0..=(words.len() - n) {
        let ngram = words[i..(i + n)].join(" ");
        ngrams.push(ngram);
    }
    
    ngrams
}

/// Calculate word frequencies
/// 
/// Calculates the frequency of each word in the text.
/// 
/// Arguments:
///     text: The input text
/// 
/// Returns:
///     A dictionary mapping words to their frequencies
#[pyfunction]
fn word_frequencies(text: &str) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    
    for word in text.split_whitespace() {
        let count = frequencies.entry(word.to_string()).or_insert(0);
        *count += 1;
    }
    
    frequencies
}

/// Calculate textual similarity metrics
/// 
/// Calculates Jaccard similarity between two texts.
/// 
/// Arguments:
///     text1: First text
///     text2: Second text
/// 
/// Returns:
///     Jaccard similarity score (0-1)
#[pyfunction]
fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
    let set1: HashSet<&str> = text1.split_whitespace().collect();
    let set2: HashSet<&str> = text2.split_whitespace().collect();
    
    let intersection_size = set1.intersection(&set2).count() as f64;
    let union_size = set1.union(&set2).count() as f64;
    
    if union_size == 0.0 {
        return 0.0;
    }
    
    intersection_size / union_size
}

/// Calculate political influence score
/// 
/// Calculates a score indicating political influence based on text content.
/// Uses a weighted approach based on frequency of political terms and phrases.
/// 
/// Arguments:
///     text: The input text
///     political_terms: List of political terms with their weights
/// 
/// Returns:
///     Political influence score (0-1)
#[pyfunction]
fn calculate_political_influence(text: &str, political_terms: HashMap<String, f64>) -> f64 {
    let lowercase_text = text.to_lowercase();
    let words: Vec<&str> = lowercase_text.split_whitespace().collect();
    
    let mut score = 0.0;
    let mut term_count = 0;
    
    for (term, weight) in political_terms.iter() {
        let term_lowercase = term.to_lowercase();
        let term_words: Vec<&str> = term_lowercase.split_whitespace().collect();
        
        if term_words.len() == 1 {
            // Single word term
            for word in &words {
                if *word == term_words[0] {
                    score += weight;
                    term_count += 1;
                }
            }
        } else {
            // Multi-word term (phrase)
            for i in 0..=words.len().saturating_sub(term_words.len()) {
                if words[i..(i + term_words.len())] == term_words[..] {
                    score += weight;
                    term_count += 1;
                }
            }
        }
    }
    
    // Normalize score to 0-1 range
    if term_count == 0 {
        return 0.0;
    }
    
    // Cap the score at 1.0
    (score / term_count as f64).min(1.0)
}

/// Calculate rhetoric intensity score
/// 
/// Calculates a score indicating the emotional intensity of rhetoric in text.
/// 
/// Arguments:
///     text: The input text
///     emotional_terms: Dictionary of emotional terms with intensity scores
/// 
/// Returns:
///     Rhetoric intensity score (0-1)
#[pyfunction]
fn calculate_rhetoric_intensity(text: &str, emotional_terms: HashMap<String, f64>) -> f64 {
    let lowercase_text = text.to_lowercase();
    let words: Vec<&str> = lowercase_text.split_whitespace().collect();
    
    let mut total_intensity = 0.0;
    let mut term_count = 0;
    
    for (term, intensity) in emotional_terms.iter() {
        let term_lowercase = term.to_lowercase();
        
        // Count occurrences of this term
        let occurrences = words.iter()
            .filter(|&word| *word == term_lowercase)
            .count();
        
        total_intensity += intensity * occurrences as f64;
        term_count += occurrences;
    }
    
    // Normalize to 0-1 range
    if term_count == 0 {
        return 0.0;
    }
    
    // Cap the score at 1.0
    (total_intensity / term_count as f64).min(1.0)
}

/// Calculate information depth score
/// 
/// Calculates a score indicating the depth of information in text.
/// Based on factors like length, vocabulary diversity, citation presence, etc.
/// 
/// Arguments:
///     text: The input text
///     word_count: Precomputed word count (optional)
/// 
/// Returns:
///     Information depth score (0-1)
#[pyfunction]
#[pyo3(signature = (text, word_count=None))]
fn calculate_information_depth(text: &str, word_count: Option<usize>) -> f64 {
    // Calculate word count if not provided
    let count = match word_count {
        Some(c) => c,
        None => text.split_whitespace().count(),
    };
    
    // Calculate unique word percentage (vocabulary diversity)
    let unique_words: HashSet<&str> = text.split_whitespace().collect();
    let unique_ratio = if count > 0 {
        unique_words.len() as f64 / count as f64
    } else {
        0.0
    };
    
    // Check for indicators of citations or references
    let has_citations = text.contains("(") && text.contains(")");
    let citation_bonus = if has_citations { 0.2 } else { 0.0 };
    
    // Calculate information density based on several factors
    let length_factor = (count as f64 / 800.0).min(1.0); // Normalize length
    let complexity_factor = unique_ratio.min(0.8); // Cap at 0.8
    
    // Combine factors into a single score (weighted sum)
    let raw_score = 0.5 * length_factor + 0.3 * complexity_factor + citation_bonus;
    
    // Ensure score is in 0-1 range
    raw_score.min(1.0).max(0.0)
}

/// Calculate all article metrics
/// 
/// Calculates all three metrics for an article in one optimized pass.
/// 
/// Arguments:
///     text: The article text
///     political_terms: Dictionary of political terms with weights
///     emotional_terms: Dictionary of emotional terms with intensities
/// 
/// Returns:
///     Tuple of (political_influence, rhetoric_intensity, information_depth)
#[pyfunction]
fn calculate_article_metrics(
    text: &str,
    political_terms: HashMap<String, f64>,
    emotional_terms: HashMap<String, f64>
) -> (f64, f64, f64) {
    // Precompute shared values
    let word_count = text.split_whitespace().count();
    
    // Calculate each metric
    let political = calculate_political_influence(text, political_terms);
    let rhetoric = calculate_rhetoric_intensity(text, emotional_terms);
    let depth = calculate_information_depth(text, Some(word_count));
    
    (political, rhetoric, depth)
}

/// MINDSET Rust acceleration module
#[pymodule]
fn mindset_rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_text, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(word_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_political_influence, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rhetoric_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_information_depth, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_article_metrics, m)?)?;
    
    Ok(())
}