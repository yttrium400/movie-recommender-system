import React, { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Typography,
  TextField,
  Box,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
} from '@mui/material';
import { Movie, MovieRecommendation, movieService, userService } from '../services/api';
import MovieCard from '../components/MovieCard';
import { useAuth } from '../contexts/AuthContext';

const Home: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [movies, setMovies] = useState<Movie[]>([]);
  const [recommendations, setRecommendations] = useState<MovieRecommendation[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [recommendationType, setRecommendationType] = useState('collaborative');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadMovies();
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      loadRecommendations();
    }
  }, [isAuthenticated, recommendationType]);

  const loadMovies = async () => {
    try {
      setLoading(true);
      const data = await movieService.getMovies(0, 20);
      setMovies(data);
    } catch (error) {
      console.error('Error loading movies:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async () => {
    try {
      setLoading(true);
      const data = await userService.getRecommendations(recommendationType);
      setRecommendations(data);
    } catch (error) {
      console.error('Error loading recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const query = event.target.value;
    setSearchQuery(query);

    if (query.length >= 2) {
      try {
        setLoading(true);
        const data = await movieService.searchMovies(query);
        setMovies(data);
      } catch (error) {
        console.error('Error searching movies:', error);
      } finally {
        setLoading(false);
      }
    } else if (query.length === 0) {
      loadMovies();
    }
  };

  const handleRating = async (movieId: number, rating: number) => {
    try {
      await userService.rateMovie({ movie_id: movieId, rating });
      // Reload recommendations after rating
      loadRecommendations();
    } catch (error) {
      console.error('Error rating movie:', error);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {isAuthenticated && (
        <>
          <Typography variant="h4" gutterBottom>
            Recommended for You
          </Typography>
          <Box sx={{ mb: 2 }}>
            <ToggleButtonGroup
              value={recommendationType}
              exclusive
              onChange={(_, value) => value && setRecommendationType(value)}
              size="small"
            >
              <ToggleButton value="collaborative">Collaborative</ToggleButton>
              <ToggleButton value="content">Content-Based</ToggleButton>
              <ToggleButton value="hybrid">Hybrid</ToggleButton>
            </ToggleButtonGroup>
          </Box>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {recommendations.map((movie) => (
              <Grid item key={movie.movie_id} xs={12} sm={6} md={4} lg={3}>
                <MovieCard
                  movie={movie}
                  onRate={(rating) => handleRating(movie.movie_id, rating)}
                  showScore
                />
              </Grid>
            ))}
          </Grid>
        </>
      )}

      <Typography variant="h4" gutterBottom>
        Browse Movies
      </Typography>
      <TextField
        fullWidth
        label="Search movies"
        variant="outlined"
        value={searchQuery}
        onChange={handleSearch}
        sx={{ mb: 3 }}
      />
      {loading ? (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {movies.map((movie) => (
            <Grid item key={movie.movie_id} xs={12} sm={6} md={4} lg={3}>
              <MovieCard
                movie={movie}
                onRate={(rating) => handleRating(movie.movie_id, rating)}
              />
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  );
};

export default Home; 