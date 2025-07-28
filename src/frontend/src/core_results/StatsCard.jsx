import React from 'react';

const StatsCard = ({ stats }) => (
    <div className="stats-card">
        <h3>Usage Statistics</h3>
        <div className="stats-grid">
            <div className="stat-item">
                <span className="stat-label">Global queries today:</span>
                <span className="stat-value">{stats.global_queries_today}/{stats.global_limit}</span>
            </div>
            <div className="stat-item">
                <span className="stat-label">Your queries this hour:</span>
                <span className="stat-value">{stats.your_queries_hour}/{stats.hourly_limit}</span>
            </div>
            <div className="stat-item">
                <span className="stat-label">Your queries today:</span>
                <span className="stat-value">{stats.your_queries_today}/{stats.daily_limit}</span>
            </div>
        </div>
        <div className="progress-bar">
            <div
                className="progress-fill"
                style={{ width: `${(stats.global_queries_today / stats.global_limit) * 100}%` }}
            ></div>
        </div>
    </div>
);
export default StatsCard;