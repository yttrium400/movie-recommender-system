import React, { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Typography,
  Box,
  Rating,
  Chip,
  CircularProgress,
  Paper,
} from '@mui/material';
import { useParams } from 'react-router-dom';
import { Movie, MovieRecommendation, movieService } from '../services/api';
import MovieCard from '../components/MovieCard';
import { useAuth } from '../contexts/AuthContext';

const MovieDetails: React.FC = () => {
  const { movieId } = useParams<{ movieId: string }>();
  const { isAuthenticated } = useAuth();
  const [movie, setMovie] = useState<Movie | null>(null);
  const [similarMovies, setSimilarMovies] = useState<MovieRecommendation[]>([]);
  const [userRating, setUserRating] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMovieData = async () => {
      if (!movieId) return;
      
      try {
        setLoading(true);
        const [movieData, similarMoviesData] = await Promise.all([
          movieService.getMovie(parseInt(movieId)),
          movieService.getSimilarMovies(parseInt(movieId))
        ]);
        
        setMovie(movieData);
        setSimilarMovies(similarMoviesData);
      } catch (error) {
        console.error('Error loading movie details:', error);
      } finally {
        setLoading(false);
      }
    };

    loadMovieData();
  }, [movieId]);

  const handleRating = async (rating: number) => {
    if (!movie) return;
    
    try {
      await movieService.rateMovie({ movie_id: movie.movie_id, rating });
      setUserRating(rating);
    } catch (error) {
      console.error('Error rating movie:', error);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" my={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (!movie) {
    return (
      <Container>
        <Typography variant="h5" color="error">Movie not found</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h4" gutterBottom>
              {movie.title}
              {movie.year && ` (${movie.year})`}
            </Typography>
            <Box sx={{ mb: 2 }}>
              {movie.genres.split('|').map((genre) => (
                <Chip
                  key={genre}
                  label={genre}
                  sx={{ mr: 1, mb: 1 }}
                />
              ))}
            </Box>
            {isAuthenticated && (
              <Box sx={{ mb: 2 }}>
                <Typography component="legend">Rate this movie</Typography>
                <Rating
                  value={userRating}
                  onChange={(_, value) => handleRating(value || 0)}
                  precision={0.5}
                />
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
        Similar Movies
      </Typography>
      <Grid container spacing={3}>
        {similarMovies.map((movie) => (
          <Grid item key={movie.movie_id} xs={12} sm={6} md={4} lg={3}>
            <MovieCard
              movie={movie}
              onRate={isAuthenticated ? handleRating : undefined}
              userRating={userRating}
              showScore
            />
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default MovieDetails; 