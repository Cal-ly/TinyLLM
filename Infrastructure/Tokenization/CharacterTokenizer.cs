using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Infrastructure.Abstractions;
using Infrastructure.Tokenization;

namespace Infrastructure.Tokenization;

/// <summary>
/// Character-level tokenizer that maps each unique character to a token ID
/// Simple, robust, and perfect for educational purposes
/// </summary>
public sealed class CharacterTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _charToToken = new();
    private readonly Dictionary<int, char> _tokenToChar = new();
    // Use a simple object for locking as the standard Monitor lock target
    private readonly object _lock = new();
    private bool _isFitted = false;

    /// <summary>
    /// Number of unique characters (tokens) in the vocabulary
    /// </summary>
    public int VocabularySize
    {
        get
        {
            lock (_lock)
            {
                return _charToToken.Count;
            }
        }
    }

    /// <summary>
    /// Whether the tokenizer has been fitted to training data
    /// </summary>
    public bool IsFitted
    {
        get
        {
            lock (_lock)
            {
                return _isFitted;
            }
        }
    }

    /// <summary>
    /// All characters in the vocabulary, sorted by token ID
    /// </summary>
    public IReadOnlyList<char> Vocabulary
    {
        get
        {
            lock (_lock)
            {
                if (!_isFitted)
                    return Array.Empty<char>();

                return _tokenToChar.OrderBy(kvp => kvp.Key)
                                  .Select(kvp => kvp.Value)
                                  .ToList()
                                  .AsReadOnly();
            }
        }
    }

    /// <summary>
    /// Build vocabulary from training text
    /// </summary>
    /// <param name="text">Training text to analyze</param>
    public void Fit(ReadOnlySpan<char> text)
    {
        if (text.IsEmpty)
            throw new ArgumentException("Training text cannot be empty");

        lock (_lock)
        {
            // Clear any existing vocabulary
            _charToToken.Clear();
            _tokenToChar.Clear();

            // Collect unique characters
            var uniqueChars = new HashSet<char>();
            foreach (char c in text)
            {
                uniqueChars.Add(c);
            }

            if (uniqueChars.Count == 0)
                throw new ArgumentException("No valid characters found in training text");

            // Assign token IDs in a deterministic order (sorted by character)
            int tokenId = 0;
            foreach (char c in uniqueChars.Order())
            {
                _charToToken[c] = tokenId;
                _tokenToChar[tokenId] = c;
                tokenId++;
            }

            _isFitted = true;
        }
    }

    /// <summary>
    /// Convert text to sequence of token IDs
    /// </summary>
    /// <param name="text">Text to tokenize</param>
    /// <returns>Array of token IDs</returns>
    public int[] Encode(ReadOnlySpan<char> text)
    {
        lock (_lock)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Tokenizer must be fitted before encoding");

            if (text.IsEmpty)
                return Array.Empty<int>();

            var tokens = new int[text.Length];
            for (int i = 0; i < text.Length; i++)
            {
                if (!_charToToken.TryGetValue(text[i], out tokens[i]))
                {
                    throw new ArgumentException($"Unknown character: '{text[i]}' (Unicode: U+{(int)text[i]:X4}). " +
                                              "This character was not seen during training. " +
                                              "Consider retraining the tokenizer with a more comprehensive dataset.");
                }
            }
            return tokens;
        }
    }

    /// <summary>
    /// Convert sequence of token IDs back to text
    /// </summary>
    /// <param name="tokens">Token IDs to decode</param>
    /// <returns>Decoded text</returns>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        lock (_lock)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Tokenizer must be fitted before decoding");

            if (tokens.IsEmpty)
                return string.Empty;

            var chars = new char[tokens.Length];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (!_tokenToChar.TryGetValue(tokens[i], out chars[i]))
                {
                    throw new ArgumentException($"Unknown token ID: {tokens[i]}. " +
                                              "Valid token IDs range from 0 to {VocabularySize - 1}.");
                }
            }
            return new string(chars);
        }
    }

    /// <summary>
    /// Convert sequence of token IDs back to text (IEnumerable version for interface compliance)
    /// </summary>
    /// <param name="tokens">Token IDs to decode</param>
    /// <returns>Decoded text</returns>
    string ITokenizer.Decode(IEnumerable<int> tokens)
    {
        lock (_lock)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Tokenizer must be fitted before decoding");

            if (tokens?.Any() != true)
                return string.Empty;

            var chars = tokens.Select(tokenId =>
            {
                if (!_tokenToChar.TryGetValue(tokenId, out char character))
                {
                    throw new ArgumentException($"Unknown token ID: {tokenId}. " +
                                                "Valid token IDs range from 0 to {VocabularySize - 1}.");
                }

                return character;
            }).ToArray();

            return new string(chars);
        }
    }

    /// <summary>
    /// Get character for a specific token ID (useful for debugging)
    /// </summary>
    /// <param name="tokenId">Token ID to lookup</param>
    /// <returns>Character corresponding to token ID</returns>
    public char GetCharacter(int tokenId)
    {
        lock (_lock)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Tokenizer must be fitted before character lookup");

            if (!_tokenToChar.TryGetValue(tokenId, out char character))
                throw new ArgumentException($"Unknown token ID: {tokenId}");

            return character;
        }
    }

    /// <summary>
    /// Get token ID for a specific character (useful for debugging)
    /// </summary>
    /// <param name="character">Character to lookup</param>
    /// <returns>Token ID corresponding to character</returns>
    public int GetTokenId(char character)
    {
        lock (_lock)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Tokenizer must be fitted before token lookup");

            if (!_charToToken.TryGetValue(character, out int tokenId))
                throw new ArgumentException($"Unknown character: '{character}'");

            return tokenId;
        }
    }

    /// <summary>
    /// Check if a character is in the vocabulary
    /// </summary>
    /// <param name="character">Character to check</param>
    /// <returns>True if character is in vocabulary</returns>
    public bool ContainsCharacter(char character)
    {
        lock (_lock)
        {
            return _isFitted && _charToToken.ContainsKey(character);
        }
    }

    /// <summary>
    /// Check if a token ID is valid
    /// </summary>
    /// <param name="tokenId">Token ID to check</param>
    /// <returns>True if token ID is valid</returns>
    public bool ContainsToken(int tokenId)
    {
        lock (_lock)
        {
            return _isFitted && _tokenToChar.ContainsKey(tokenId);
        }
    }

    /// <summary>
    /// Get detailed information about the vocabulary (useful for debugging)
    /// </summary>
    /// <returns>Vocabulary information</returns>
    public VocabularyInfo GetVocabularyInfo()
    {
        lock (_lock)
        {
            if (!_isFitted)
                return new VocabularyInfo(0, Array.Empty<CharacterInfo>());

            var characterInfos = _charToToken
                .OrderBy(kvp => kvp.Value)
                .Select(kvp => new CharacterInfo(
                    Character: kvp.Key,
                    TokenId: kvp.Value,
                    UnicodeCategory: char.GetUnicodeCategory(kvp.Key),
                    IsControl: char.IsControl(kvp.Key),
                    IsWhiteSpace: char.IsWhiteSpace(kvp.Key),
                    DisplayName: GetCharacterDisplayName(kvp.Key)
                ))
                .ToArray();

            return new VocabularyInfo(_charToToken.Count, characterInfos);
        }
    }

    /// <summary>
    /// Get current tokenizer state for serialization
    /// </summary>
    /// <returns>Tokenizer state</returns>
    public TokenizerState GetState()
    {
        lock (_lock)
        {
            return new TokenizerState
            {
                CharToToken = new Dictionary<char, int>(_charToToken),
                TokenToChar = new Dictionary<int, char>(_tokenToChar),
                IsFitted = _isFitted
            };
        }
    }

    /// <summary>
    /// Load tokenizer state from serialization
    /// </summary>
    /// <param name="state">Tokenizer state to load</param>
    public void LoadState(TokenizerState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        lock (_lock)
        {
            _charToToken.Clear();
            _tokenToChar.Clear();

            // Validate state consistency
            if (state.CharToToken.Count != state.TokenToChar.Count)
                throw new ArgumentException("Inconsistent tokenizer state: character and token dictionaries have different sizes");

            // Validate bidirectional mapping
            foreach (var (character, tokenId) in state.CharToToken)
            {
                if (!state.TokenToChar.TryGetValue(tokenId, out char mappedChar) || mappedChar != character)
                    throw new ArgumentException($"Inconsistent tokenizer state: character '{character}' maps to token {tokenId}, but token {tokenId} doesn't map back to the same character");
            }

            // Load the mappings
            foreach (var kvp in state.CharToToken)
                _charToToken[kvp.Key] = kvp.Value;

            foreach (var kvp in state.TokenToChar)
                _tokenToChar[kvp.Key] = kvp.Value;

            _isFitted = state.IsFitted;
        }
    }

    /// <summary>
    /// Reset tokenizer to unfitted state
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _charToToken.Clear();
            _tokenToChar.Clear();
            _isFitted = false;
        }
    }

    /// <summary>
    /// Create a user-friendly display name for a character
    /// </summary>
    private static string GetCharacterDisplayName(char character)
    {
        return character switch
        {
            '\n' => "\\n (newline)",
            '\r' => "\\r (carriage return)",
            '\t' => "\\t (tab)",
            ' ' => "' ' (space)",
            _ when char.IsControl(character) => $"U+{(int)character:X4} (control)",
            _ when char.IsWhiteSpace(character) => $"'{character}' (whitespace)",
            _ => $"'{character}'"
        };
    }
}