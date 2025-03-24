import React, { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Typography,
  Box,
  Paper,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
} from '@mui/material';
import { Movie, MovieRating, userService } from '../services/api';
import MovieCard from '../components/MovieCard';
import { useAuth } from '../contexts/AuthContext';
import { Navigate } from 'react-router-dom';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`profile-tabpanel-${index}`}
      aria-labelledby={`profile-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

interface UserStats {
  totalRatings: number;
  averageRating: number;
  favoriteGenres: { genre: string; count: number }[];
}

const UserProfile: React.FC = () => {
  const { isAuthenticated, user } = useAuth();
  const [tabValue, setTabValue] = useState(0);
  const [ratedMovies, setRatedMovies] = useState<(Movie & { rating: number })[]>([]);
  const [recommendations, setRecommendations] = useState<Movie[]>([]);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadUserData = async () => {
      try {
        setLoading(true);
        const [ratingsData, recommendationsData] = await Promise.all([
          userService.getRatedMovies(),
          userService.getRecommendations('hybrid', 8)
        ]);

        setRatedMovies(ratingsData);
        setRecommendations(recommendationsData);

        // Calculate user stats
        const totalRatings = ratingsData.length;
        const averageRating = ratingsData.reduce((acc, curr) => acc + curr.rating, 0) / totalRatings;
        
        // Count genres
        const genreCounts: { [key: string]: number } = {};
        ratingsData.forEach(movie => {
          movie.genres.split('|').forEach(genre => {
            genreCounts[genre] = (genreCounts[genre] || 0) + 1;
          });
        });

        // Get top 5 genres
        const favoriteGenres = Object.entries(genreCounts)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([genre, count]) => ({ genre, count }));

        setStats({
          totalRatings,
          averageRating,
          favoriteGenres
        });
      } catch (error) {
        console.error('Error loading user data:', error);
      } finally {
        setLoading(false);
      }
    };

    if (isAuthenticated) {
      loadUserData();
    }
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" my={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ mb: 4 }}>
        <Box sx={{ p: 3 }}>
          <Typography variant="h4" gutterBottom>
            {user?.username}'s Profile
          </Typography>
          {stats && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6">Total Ratings</Typography>
                <Typography variant="h4">{stats.totalRatings}</Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6">Average Rating</Typography>
                <Typography variant="h4">
                  {stats.averageRating.toFixed(1)}
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="h6">Top Genres</Typography>
                <Box>
                  {stats.favoriteGenres.map(({ genre, count }) => (
                    <Typography key={genre} variant="body1">
                      {genre}: {count} movies
                    </Typography>
                  ))}
                </Box>
              </Grid>
            </Grid>
          )}
        </Box>
      </Paper>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          aria-label="profile tabs"
        >
          <Tab label="Rated Movies" />
          <Tab label="Recommended" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {ratedMovies.map((movie) => (
            <Grid item key={movie.movie_id} xs={12} sm={6} md={4} lg={3}>
              <MovieCard
                movie={movie}
                userRating={movie.rating}
              />
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          {recommendations.map((movie) => (
            <Grid item key={movie.movie_id} xs={12} sm={6} md={4} lg={3}>
              <MovieCard movie={movie} />
            </Grid>
          ))}
        </Grid>
      </TabPanel>
    </Container>
  );
};

export default UserProfile; 