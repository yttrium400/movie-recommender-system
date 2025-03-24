import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Rating,
  Chip,
  Box,
  CardActionArea,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { Movie } from '../services/api';

interface MovieCardProps {
  movie: Movie & { score?: number };
  onRate?: (rating: number) => void;
  userRating?: number;
  showScore?: boolean;
}

const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  onRate,
  userRating,
  showScore = false,
}) => {
  const navigate = useNavigate();
  const genres = movie.genres.split('|');

  const handleClick = () => {
    navigate(`/movie/${movie.movie_id}`);
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardActionArea onClick={handleClick}>
        <CardContent sx={{ flexGrow: 1 }}>
          <Typography gutterBottom variant="h6" component="div">
            {movie.title}
            {movie.year && ` (${movie.year})`}
          </Typography>
          <Box sx={{ mb: 1 }}>
            {genres.map((genre) => (
              <Chip
                key={genre}
                label={genre}
                size="small"
                sx={{ mr: 0.5, mb: 0.5 }}
              />
            ))}
          </Box>
          {showScore && movie.score !== undefined && (
            <Typography variant="body2" color="text.secondary">
              Similarity Score: {(movie.score * 100).toFixed(1)}%
            </Typography>
          )}
        </CardContent>
      </CardActionArea>
      {onRate && (
        <CardActions>
          <Rating
            value={userRating || 0}
            onChange={(_, value) => onRate(value || 0)}
            precision={0.5}
          />
        </CardActions>
      )}
    </Card>
  );
};

export default MovieCard; 