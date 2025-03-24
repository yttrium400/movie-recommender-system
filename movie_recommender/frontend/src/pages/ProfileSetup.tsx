import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Container, 
  TextField, 
  Typography, 
  Paper, 
  Chip, 
  Grid, 
  Alert, 
  CircularProgress,
  LinearProgress
} from '@mui/material';
import { userService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const ProfileSetup = () => {
  const navigate = useNavigate();
  const { user, updateUser } = useAuth();
  const [fullName, setFullName] = useState('');
  const [genres, setGenres] = useState<string[]>([]);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingGenres, setLoadingGenres] = useState(true);

  useEffect(() => {
    const loadGenres = async () => {
      try {
        const availableGenres = await userService.getGenres();
        setGenres(availableGenres);
      } catch (err) {
        setError('Failed to load genres. Please try again.');
      } finally {
        setLoadingGenres(false);
      }
    };

    loadGenres();
  }, []);

  const handleGenreClick = (genre: string) => {
    // If already selected, remove it
    if (selectedGenres.includes(genre)) {
      setSelectedGenres(selectedGenres.filter(g => g !== genre));
    } 
    // If not selected and fewer than 5 genres are selected, add it
    else if (selectedGenres.length < 5) {
      setSelectedGenres([...selectedGenres, genre]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (selectedGenres.length !== 5) {
      setError('Please select exactly 5 favorite genres.');
      return;
    }

    try {
      setLoading(true);
      const updatedProfile = await userService.updateUserProfile({
        full_name: fullName,
        preferred_genres: selectedGenres
      });

      // Update user in context
      updateUser(updatedProfile);
      
      // Navigate to home page
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update profile. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loadingGenres) {
    return (
      <Container maxWidth="sm">
        <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Setting Up Your Profile
          </Typography>
          <LinearProgress />
          <Typography variant="body1" sx={{ mt: 2 }} align="center">
            Loading genres...
          </Typography>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Complete Your Profile
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box component="form" onSubmit={handleSubmit} noValidate>
          <TextField
            margin="normal"
            fullWidth
            id="fullName"
            label="Full Name (Optional)"
            name="fullName"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
          />
          
          <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
            Select 5 of your favorite movie genres:
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Selected: {selectedGenres.length}/5
          </Typography>
          
          <Grid container spacing={1}>
            {genres.map(genre => (
              <Grid item key={genre}>
                <Chip
                  label={genre}
                  onClick={() => handleGenreClick(genre)}
                  color={selectedGenres.includes(genre) ? "primary" : "default"}
                  variant={selectedGenres.includes(genre) ? "filled" : "outlined"}
                  sx={{ m: 0.5 }}
                />
              </Grid>
            ))}
          </Grid>
          
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 4, mb: 2 }}
            disabled={loading || selectedGenres.length !== 5}
          >
            {loading ? <CircularProgress size={24} /> : 'Finish Your Profile'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default ProfileSetup; 