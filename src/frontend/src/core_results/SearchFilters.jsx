import React from 'react';

const SearchFilters = ({ filters, filterOptions, onFilterChange }) => (
    <div className="filters-section">
        <h4>Filters</h4>
        <div className="filters-grid">
            <div className="filter-group">
                <label>Language:</label>
                <select
                    value={filters.language || ''}
                    onChange={(e) => onFilterChange('language', e.target.value || null)}
                >
                    <option value="">Any</option>
                    {filterOptions.languages?.map(lang => (
                        <option key={lang} value={lang}>{lang}</option>
                    ))}
                </select>
            </div>

            <div className="filter-group">
                <label>Period:</label>
                <select
                    value={filters.period || ''}
                    onChange={(e) => onFilterChange('period', e.target.value || null)}
                >
                    <option value="">Any</option>
                    <option value="early_medieval">Early Medieval (10-12th c.)</option>
                    <option value="late_medieval">Late Medieval (13-15th c.)</option>
                    <option value="early_modern">Early Modern (16-18th c.)</option>
                </select>
            </div>

            <div className="filter-group">
                <label>Document Type:</label>
                <select
                    value={filters.document_type || ''}
                    onChange={(e) => onFilterChange('document_type', e.target.value || null)}
                >
                    <option value="">Any</option>
                    {filterOptions.document_types?.map(type => (
                        <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
                    ))}
                </select>
            </div>

            <div className="filter-group">
                <label>Institution:</label>
                <select
                    value={filters.institution || ''}
                    onChange={(e) => onFilterChange('institution', e.target.value || null)}
                >
                    <option value="">Any</option>
                    {filterOptions.institutions?.map(inst => (
                        <option key={inst} value={inst}>{inst.charAt(0).toUpperCase() + inst.slice(1)}</option>
                    ))}
                </select>
            </div>

            <div className="filter-group">
                <label>Collection:</label>
                <select
                    value={filters.collection || ''}
                    onChange={(e) => onFilterChange('collection', e.target.value || null)}
                >
                    <option value="">Any</option>
                    {filterOptions.collections?.map(coll => (
                        <option key={coll} value={coll}>
                            {coll.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </option>
                    ))}
                </select>
            </div>

            <div className="filter-group">
                <label>Content:</label>
                <div className="checkbox-group">
                    <label className="checkbox-label">
                        <input
                            type="checkbox"
                            checked={filters.has_transcriptions === true}
                            onChange={(e) => onFilterChange('has_transcriptions', e.target.checked ? true : null)}
                        />
                        Has Transcriptions
                    </label>
                    <label className="checkbox-label">
                        <input
                            type="checkbox"
                            checked={filters.has_translations === true}
                            onChange={(e) => onFilterChange('has_translations', e.target.checked ? true : null)}
                        />
                        Has Translations
                    </label>
                </div>
            </div>
        </div>
    </div>
);

export default SearchFilters;